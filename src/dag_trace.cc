/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * DAG trace infrastructure – only compiled when NCCL_DAG_TRACE_ENABLED=1.
 *
 * Architecture
 * ────────────
 *  • Each thread that calls ncclDagEmitNode() gets a thread-local SPSC ring
 *    buffer (DagRingBuffer).
 *  • A single low-priority drain thread periodically sweeps all registered
 *    ring buffers and serialises nodes into a newline-delimited JSON file
 *    (one file per process: nccl_dag_<pid>.jsonl).
 *  • The ring buffers are lock-free (single-producer / single-consumer).
 *    If a buffer is full the newest events are dropped (never block).
 */

#include "dag_trace.h"

#if NCCL_DAG_TRACE_ENABLED

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cerrno>

/*
 * When compiled as part of the full NCCL build, param.h and debug.h provide
 * NCCL_PARAM() and INFO().  For standalone testing we provide minimal stubs.
 */
#ifdef NCCL_DAG_TRACE_STANDALONE
  #define NCCL_PARAM(name, env, deftVal) \
    static int64_t ncclParam##name() { \
      const char* v = getenv("NCCL_" env); \
      return v ? atoll(v) : (int64_t)(deftVal); \
    }
  static const char* ncclGetEnv(const char* name) { return getenv(name); }
  #define INFO(flags, ...) do { fprintf(stderr, "INFO: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
  #define NCCL_ALL 0
#else
  #include "param.h"
  #include "debug.h"
#endif
#include <cstring>
#include <cinttypes>
#include <cerrno>

#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <string>

#include <time.h>
#include <unistd.h>
#include <sys/types.h>

/* ── runtime parameter (NCCL_DAG_TRACE=0|1) ────────────────────────── */
NCCL_PARAM(DagTrace, "DAG_TRACE", 0);

/* Global runtime switch. */
std::atomic<bool> ncclDagTraceActive{false};

/* ── SPSC ring buffer ──────────────────────────────────────────────── */

static constexpr size_t DAG_RING_CAPACITY = 1u << 16; /* 65 536 slots */

struct DagRingBuffer {
  alignas(64) std::atomic<uint64_t> writePos{0};
  alignas(64) uint64_t              readPos{0};
  DagNode nodes[DAG_RING_CAPACITY];

  bool tryPush(const DagNode& n) {
    uint64_t wp = writePos.load(std::memory_order_relaxed);
    if (wp - readPos >= DAG_RING_CAPACITY) return false;   /* full → drop */
    nodes[wp & (DAG_RING_CAPACITY - 1)] = n;
    writePos.store(wp + 1, std::memory_order_release);
    return true;
  }

  bool tryPop(DagNode& out) {
    uint64_t wp = writePos.load(std::memory_order_acquire);
    if (readPos >= wp) return false;
    out = nodes[readPos & (DAG_RING_CAPACITY - 1)];
    readPos++;
    return true;
  }
};

/* ── per-thread ring and global registry ───────────────────────────── */

static thread_local DagRingBuffer* tl_ring = nullptr;

static std::mutex              g_registryMtx;
static std::vector<DagRingBuffer*> g_rings;   /* guarded by g_registryMtx */

static DagRingBuffer* dagGetRing() {
  if (__builtin_expect(tl_ring != nullptr, 1)) return tl_ring;
  tl_ring = new DagRingBuffer();
  {
    std::lock_guard<std::mutex> lk(g_registryMtx);
    g_rings.push_back(tl_ring);
  }
  return tl_ring;
}

/* ── global node-id counter ────────────────────────────────────────── */

static std::atomic<uint64_t> g_nodeIdCounter{0};

uint64_t ncclDagAllocNodeId() {
  return g_nodeIdCounter.fetch_add(1, std::memory_order_relaxed);
}

/* ── timestamp ─────────────────────────────────────────────────────── */

uint64_t ncclDagTimestampNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── emit helpers ──────────────────────────────────────────────────── */

void ncclDagEmitNode(const DagNode& node) {
  DagRingBuffer* ring = dagGetRing();
  ring->tryPush(node);   /* drop on overflow – never block */
}

uint64_t ncclDagEmit(DagLayer layer, DagEventType evType,
                     uint64_t parentNodeId,
                     uint64_t opCount, uint32_t commHash,
                     int32_t channelId, int32_t peer,
                     uint8_t protocol, uint8_t algorithm,
                     uint64_t bytes, const char* detail,
                     const uint64_t* deps, uint8_t numDeps) {
  DagNode n;
  memset(&n, 0, sizeof(n));
  n.nodeId       = ncclDagAllocNodeId();
  n.timestampNs  = ncclDagTimestampNs();
  n.layerId      = (uint16_t)layer;
  n.eventType    = (uint16_t)evType;
  n.commHash     = commHash;
  n.opCount      = opCount;
  n.channelId    = channelId;
  n.peer         = peer;
  n.protocol     = protocol;
  n.algorithm    = algorithm;
  n.bytes        = bytes;
  n.parentNodeId = parentNodeId;
  if (numDeps > DAG_MAX_DEPS) numDeps = DAG_MAX_DEPS;
  n.numDeps      = numDeps;
  for (uint8_t i = 0; i < numDeps; i++) n.dependsOn[i] = deps[i];
  if (detail) {
    strncpy(n.detail, detail, sizeof(n.detail) - 1);
    n.detail[sizeof(n.detail) - 1] = '\0';
  }
  ncclDagEmitNode(n);
  return n.nodeId;
}

/* ── JSON serialisation (one node → one line) ──────────────────────── */

static const char* dagLayerStr(uint16_t l) {
  switch (l) {
    case DagLayerAPI:     return "api";
    case DagLayerKernel:  return "kernel";
    case DagLayerProxy:   return "proxy";
    case DagLayerNetwork: return "network";
    default:              return "unknown";
  }
}

static const char* dagEventStr(uint16_t e) {
  switch (e) {
    case DagEvGroupStart:    return "group_start";
    case DagEvGroupEnd:      return "group_end";
    case DagEvCollEnqueue:   return "coll_enqueue";
    case DagEvKernelLaunch:  return "kernel_launch";
    case DagEvProxyOpBegin:  return "proxy_op_begin";
    case DagEvProxyOpEnd:    return "proxy_op_end";
    case DagEvProxySendPost: return "proxy_send_post";
    case DagEvProxySendXmit: return "proxy_send_xmit";
    case DagEvProxySendDone: return "proxy_send_done";
    case DagEvProxyRecvPost: return "proxy_recv_post";
    case DagEvProxyRecvRecv: return "proxy_recv_recv";
    case DagEvProxyRecvXmit: return "proxy_recv_xmit";
    case DagEvProxyRecvDone: return "proxy_recv_done";
    case DagEvNetPostSend:   return "net_post_send";
    case DagEvNetPollCqDone: return "net_poll_cq_done";
    case DagEvNetPostRecv:   return "net_post_recv";
    default:                 return "unknown";
  }
}

static void dagWriteNode(FILE* fp, const DagNode& n) {
  /* deps array */
  char depsBuf[256];
  depsBuf[0] = '\0';
  int pos = 0;
  for (uint8_t i = 0; i < n.numDeps && i < DAG_MAX_DEPS; i++) {
    if (i > 0) pos += snprintf(depsBuf + pos, sizeof(depsBuf) - pos, ",");
    pos += snprintf(depsBuf + pos, sizeof(depsBuf) - pos, "%" PRIu64, n.dependsOn[i]);
  }

  fprintf(fp,
    "{\"id\":%" PRIu64 ",\"ts\":%" PRIu64 ","
    "\"layer\":\"%s\",\"event\":\"%s\","
    "\"commHash\":%u,\"opCount\":%" PRIu64 ","
    "\"ch\":%d,\"peer\":%d,"
    "\"proto\":%u,\"algo\":%u,"
    "\"parent\":%" PRIu64 ",\"deps\":[%s],"
    "\"bytes\":%" PRIu64 ",\"detail\":\"%s\"}\n",
    n.nodeId, n.timestampNs,
    dagLayerStr(n.layerId), dagEventStr(n.eventType),
    n.commHash, n.opCount,
    n.channelId, n.peer,
    (unsigned)n.protocol, (unsigned)n.algorithm,
    n.parentNodeId, depsBuf,
    n.bytes, n.detail);
}

/* ── drain thread ──────────────────────────────────────────────────── */

static std::thread             g_drainThread;
static std::atomic<bool>       g_drainStop{false};
static FILE*                   g_outFile = nullptr;

static void dagDrainOnce() {
  if (!g_outFile) return;
  DagNode n;
  std::vector<DagRingBuffer*> snapshot;
  {
    std::lock_guard<std::mutex> lk(g_registryMtx);
    snapshot = g_rings;
  }
  for (DagRingBuffer* ring : snapshot) {
    while (ring->tryPop(n)) {
      dagWriteNode(g_outFile, n);
    }
  }
  fflush(g_outFile);
}

static void dagDrainLoop() {
  while (!g_drainStop.load(std::memory_order_relaxed)) {
    dagDrainOnce();
    /* sleep 10 ms between sweeps – low overhead, acceptable latency */
    struct timespec req = {0, 10000000};
    nanosleep(&req, nullptr);
  }
  /* final sweep */
  dagDrainOnce();
}

/* ── public init / finalize ────────────────────────────────────────── */

void ncclDagTraceInit() {
  if (ncclParamDagTrace() == 0) {
    ncclDagTraceActive.store(false, std::memory_order_relaxed);
    return;
  }

  /* open output file */
  char path[256];
  const char* dir = ncclGetEnv("NCCL_DAG_TRACE_DIR");
  if (!dir || dir[0] == '\0') dir = "/tmp";
  snprintf(path, sizeof(path), "%s/nccl_dag_%d.jsonl", dir, (int)getpid());

  g_outFile = fopen(path, "w");
  if (!g_outFile) {
    INFO(NCCL_ALL, "DAG trace: cannot open %s (%s) – tracing disabled", path, strerror(errno));
    ncclDagTraceActive.store(false, std::memory_order_relaxed);
    return;
  }

  g_drainStop.store(false, std::memory_order_relaxed);
  g_drainThread = std::thread(dagDrainLoop);

  ncclDagTraceActive.store(true, std::memory_order_release);

  /* Register atexit so the drain thread is cleaned up even if no
   * explicit ncclDagTraceFinalize() is called. */
  atexit(ncclDagTraceFinalize);

  INFO(NCCL_ALL, "DAG trace: enabled, output → %s", path);
}

void ncclDagTraceFinalize() {
  if (!ncclDagTraceActive.load(std::memory_order_relaxed)) return;

  ncclDagTraceActive.store(false, std::memory_order_release);

  /* stop drain thread */
  g_drainStop.store(true, std::memory_order_release);
  if (g_drainThread.joinable()) g_drainThread.join();

  if (g_outFile) {
    fclose(g_outFile);
    g_outFile = nullptr;
  }
  INFO(NCCL_ALL, "DAG trace: finalized");
}

void ncclDagFlush() {
  dagDrainOnce();
}

#endif /* NCCL_DAG_TRACE_ENABLED */
