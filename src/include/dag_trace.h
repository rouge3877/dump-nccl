/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DAG_TRACE_H_
#define NCCL_DAG_TRACE_H_

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstring>

/*
 * Compile-time master switch: build with -DNCCL_DAG_TRACE_ENABLED=1 to include
 * the tracing code paths.  When set to 0 (the default) every DAG_TRACE_*
 * macro expands to nothing, giving true zero overhead.
 */
#ifndef NCCL_DAG_TRACE_ENABLED
#define NCCL_DAG_TRACE_ENABLED 0
#endif

/* ── node / edge types ─────────────────────────────────────────────── */

enum DagLayer : uint16_t {
  DagLayerAPI      = 0,   /* ncclAllReduce, ncclGroupStart/End, etc. */
  DagLayerKernel   = 1,   /* kernel launch / kernel channel */
  DagLayerProxy    = 2,   /* proxy op / proxy step */
  DagLayerNetwork  = 3,   /* ibv_post_send, ibv_poll_cq, etc. */
};

enum DagEventType : uint16_t {
  /* Layer 0 – API */
  DagEvGroupStart    = 0,
  DagEvGroupEnd      = 1,
  DagEvCollEnqueue   = 2,

  /* Layer 1 – Kernel */
  DagEvKernelLaunch  = 10,

  /* Layer 2 – Proxy */
  DagEvProxyOpBegin  = 20,
  DagEvProxyOpEnd    = 21,
  DagEvProxySendPost = 22,
  DagEvProxySendXmit = 23,
  DagEvProxySendDone = 24,
  DagEvProxyRecvPost = 25,
  DagEvProxyRecvRecv = 26,
  DagEvProxyRecvXmit = 27,
  DagEvProxyRecvDone = 28,

  /* Layer 3 – Network */
  DagEvNetPostSend   = 40,
  DagEvNetPollCqDone = 41,
  DagEvNetPostRecv   = 42,
};

/* ── maximum number of dependency edges per node ───────────────────── */
#define DAG_MAX_DEPS 4

/* ── invalid node sentinel ─────────────────────────────────────────── */
#define DAG_INVALID_NODE UINT64_MAX

/* ── DagNode: the fundamental trace record ─────────────────────────── */
struct DagNode {
  uint64_t    nodeId;                 /* global monotonic id */
  uint64_t    timestampNs;            /* CLOCK_MONOTONIC_RAW ns */
  uint16_t    layerId;                /* DagLayer */
  uint16_t    eventType;              /* DagEventType */
  uint32_t    commHash;               /* communicator hash (lower 32 bits) */
  uint64_t    opCount;                /* operation counter */
  int32_t     channelId;              /* -1 if N/A */
  int32_t     peer;                   /* peer rank, -1 if N/A */
  uint8_t     protocol;               /* NCCL_PROTO_* */
  uint8_t     algorithm;              /* NCCL_ALGO_* */
  uint8_t     numDeps;                /* how many entries in dependsOn[] */
  uint8_t     pad_;
  uint64_t    parentNodeId;           /* direct causal parent */
  uint64_t    dependsOn[DAG_MAX_DEPS];
  uint64_t    bytes;                  /* data volume */
  char        detail[32];             /* short description */
};

/* ── compile-time-disabled stubs ───────────────────────────────────── */

#if NCCL_DAG_TRACE_ENABLED

/*
 * Runtime flag – even when the code is compiled in, the user can leave
 * tracing dormant (NCCL_DAG_TRACE=0).  We read with relaxed ordering and
 * hide behind __builtin_expect to keep the branch predictor happy on the
 * fast path.
 */
extern std::atomic<bool> ncclDagTraceActive;

/* Allocate / free the global drain infrastructure (called once). */
void ncclDagTraceInit(void);
void ncclDagTraceFinalize(void);

/* Return a process-wide unique node id. */
uint64_t ncclDagAllocNodeId(void);

/* Get wall-clock in nanoseconds (CLOCK_MONOTONIC_RAW). */
uint64_t ncclDagTimestampNs(void);

/* Push a DagNode into the per-thread ring buffer (non-blocking, lossy). */
void ncclDagEmitNode(const DagNode& node);

/*
 * Convenience: build a DagNode, emit it, and return its nodeId.
 * The variadic helper avoids repetitive boiler-plate at call sites.
 */
uint64_t ncclDagEmit(DagLayer layer, DagEventType evType,
                     uint64_t parentNodeId,
                     uint64_t opCount, uint32_t commHash,
                     int32_t channelId, int32_t peer,
                     uint8_t protocol, uint8_t algorithm,
                     uint64_t bytes, const char* detail,
                     const uint64_t* deps = nullptr, uint8_t numDeps = 0);

/* Flush all per-thread ring buffers to disk (drain thread does this
 * periodically; explicit call from finalize path). */
void ncclDagFlush(void);

/* ── macros ─────────────────────────────────────────────────────────── */

#define DAG_TRACE_IF(expr) \
  do { \
    if (__builtin_expect(ncclDagTraceActive.load(std::memory_order_relaxed), 0)) { \
      expr; \
    } \
  } while (0)

#define DAG_TRACE_ACTIVE() \
  (__builtin_expect(ncclDagTraceActive.load(std::memory_order_relaxed), 0))

#else /* NCCL_DAG_TRACE_ENABLED == 0  ── everything compiles away ──── */

static inline void ncclDagTraceInit(void) {}
static inline void ncclDagTraceFinalize(void) {}
static inline uint64_t ncclDagAllocNodeId(void) { return DAG_INVALID_NODE; }
static inline uint64_t ncclDagTimestampNs(void) { return 0; }

#define DAG_TRACE_IF(expr) ((void)0)
#define DAG_TRACE_ACTIVE() (false)

#endif /* NCCL_DAG_TRACE_ENABLED */

#endif /* NCCL_DAG_TRACE_H_ */
