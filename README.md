# NCCL

Optimized primitives for inter-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Build

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## Install

To install NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for NCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## DAG Trace — Operation Execution DAG Export

This fork adds a **zero-overhead DAG tracing mechanism** that can export the full execution DAG (Directed Acyclic Graph) of NCCL operations for performance analysis and observability. The trace captures causal dependencies across four abstraction layers, from the user-facing API down to IB verbs.

### Architecture

NCCL's internal execution is modeled as a 4-layer hierarchy. The DAG tracer instruments each layer:

```
Layer 0 ─ API / Scheduling    ncclGroupStart → ncclAllReduce → ncclGroupEnd
              │                    ncclEnqueueCheck, ncclLaunchKernel
              ▼
Layer 1 ─ Kernel Launch        CUDA kernel dispatch per channel
              │                    ncclKernelMain, work batch loading
              ▼
Layer 2 ─ Proxy Thread         Host-side data pump (proxy op state machine)
              │                    posted → transmitted → done (send)
              │                    posted → received → transmitted → done (recv)
              ▼
Layer 3 ─ Network / IB Verbs   ibv_post_send, ibv_poll_cq, ibv_irecv
```

Each DAG node records: timestamp, layer, event type, `commHash`, `opCount`, `channelId`, `peer`, `protocol`, `algorithm`, data size, and a `parentNodeId` linking to its causal predecessor. Cross-layer linkage is carried via `dagNodeId` fields threaded through `ncclKernelPlan` → `ncclProxyOp` → `ncclProxySubArgs`.

### Zero-Overhead Guarantee

The tracing is gated by a **compile-time** switch (`NCCL_DAG_TRACE_ENABLED`). When disabled (the default), every `DAG_TRACE_IF()` macro expands to `((void)0)` — the compiler eliminates all trace code entirely. Verified at the assembly level:

```asm
; foo() with DAG_TRACE_IF disabled compiles to:
_Z3foov:
    xorl  %eax, %eax
    ret
```

When compiled in, a **runtime** switch (`NCCL_DAG_TRACE` env var) controls activation. The runtime check uses `__builtin_expect(..., 0)` so the branch predictor favors the non-tracing path.

### Building with DAG Trace

**Make:**
```shell
$ make -j src.build DAG_TRACE=1
```

**CMake:**
```shell
$ cmake -DNCCL_DAG_TRACE=ON ...
```

Without `DAG_TRACE=1`, the library is identical to upstream — zero binary size increase, zero performance impact.

### Collecting Traces

```shell
# Enable at runtime (requires DAG_TRACE=1 at build time)
$ export NCCL_DAG_TRACE=1
$ export NCCL_DAG_TRACE_DIR=/tmp    # output directory (default: /tmp)

# Run your application
$ mpirun -np 4 ./my_nccl_app

# One JSONL file per process is created:
$ ls /tmp/nccl_dag_*.jsonl
/tmp/nccl_dag_12345.jsonl
/tmp/nccl_dag_12346.jsonl
...
```

Each line in the JSONL file is a self-contained JSON object representing one DAG node:
```json
{"id":3,"ts":1713900000000,"layer":"proxy","event":"proxy_op_begin",
 "commHash":48879,"opCount":1,"ch":0,"peer":3,"proto":2,"algo":1,
 "parent":2,"deps":[],"bytes":4096,"detail":"ProxyOpBegin"}
```

### Analyzing Traces

The `tools/nccl_dag_merge.py` script merges multi-rank traces and provides several output formats:

```shell
# Summary statistics
$ python3 tools/nccl_dag_merge.py nccl_dag_*.jsonl --summary

# Chrome Trace (open in chrome://tracing or Perfetto)
$ python3 tools/nccl_dag_merge.py nccl_dag_*.jsonl --chrome-trace trace.json

# Graphviz DOT graph
$ python3 tools/nccl_dag_merge.py nccl_dag_*.jsonl --dot graph.dot
$ dot -Tpng graph.dot -o graph.png

# Merged JSON with cross-rank edges
$ python3 tools/nccl_dag_merge.py nccl_dag_*.jsonl -o merged.json

# Critical path analysis
$ python3 tools/nccl_dag_merge.py nccl_dag_*.jsonl --critical-path
```

Cross-rank edges are automatically detected by matching `proxy_send_xmit` events on one rank with `proxy_recv_recv` events on the peer rank using `(opCount, channelId)` as the correlation key.

### Instrumented Events

| Layer | Event | Injection Point | Description |
|-------|-------|----------------|-------------|
| API | `group_start` | `ncclGroupStart()` | Group operation begins |
| API | `group_end` | `ncclGroupEnd()` | Group operation completes |
| API | `coll_enqueue` | `ncclEnqueueCheck()` | Collective enqueued (records op name, data size) |
| Kernel | `kernel_launch` | `ncclLaunchKernel()` | CUDA kernel dispatched |
| Proxy | `proxy_op_begin` | `ncclProxyOpToArgs()` | Proxy operation starts (records channel, peer) |
| Proxy | `proxy_op_end` | `removeOp()` | Proxy operation completes |
| Proxy | `proxy_send_post` | `sendProxyProgress()` | GPU buffer posted for send |
| Proxy | `proxy_send_xmit` | `sendProxyProgress()` | Data transmitted via `ncclNet->isend` |
| Proxy | `proxy_send_done` | `sendProxyProgress()` | Network send completed |
| Proxy | `proxy_recv_post` | `recvProxyProgress()` | `irecv` posted to network |
| Proxy | `proxy_recv_recv` | `recvProxyProgress()` | Data received from network |
| Proxy | `proxy_recv_xmit` | `recvProxyProgress()` | Data transmitted to GPU |
| Proxy | `proxy_recv_done` | `recvProxyProgress()` | GPU acknowledged completion |
| Network | `net_post_send` | `ncclIbMultiSend()` | `ibv_post_send` issued |
| Network | `net_poll_cq_done` | `ncclIbTest()` | CQ completion polled |
| Network | `net_post_recv` | `ncclIbIrecv()` | Receive posted to IB |

### Implementation Details

**Data path (hot path is lock-free):**
1. `DAG_TRACE_IF` checks `ncclDagTraceActive` (relaxed atomic load, branch-predicted cold)
2. `ncclDagEmit()` fills a `DagNode` (128 bytes) with timestamp + metadata
3. Node is pushed to a **thread-local SPSC ring buffer** (64K slots) — no locks, no allocation
4. A background **drain thread** (10ms sweep interval) pops nodes and writes JSONL to a file
5. On overflow, newest events are dropped (never blocks the producer)

**Files changed:**

| File | Change |
|------|--------|
| `src/include/dag_trace.h` | Core header: `DagNode`, enums, macros, API declarations |
| `src/dag_trace.cc` | Ring buffer, drain thread, init/finalize, JSONL serialization |
| `makefiles/common.mk` | `DAG_TRACE` build variable |
| `src/CMakeLists.txt` | `NCCL_DAG_TRACE` CMake option |
| `src/Makefile` | `dag_trace.cc` added to source list |
| `src/init.cc` | `ncclDagTraceInit()` via `std::call_once` |
| `src/include/comm.h` | `dagNodeId` field in `ncclKernelPlan` |
| `src/include/info.h` | `dagNodeId` field in `ncclInfo` |
| `src/include/proxy.h` | `dagParentNodeId` in `ncclProxyOp`; `dagOpNodeId`, `dagParentNodeId` in `ncclProxySubArgs` |
| `src/group.cc` | `DagEvGroupStart` / `DagEvGroupEnd` emission |
| `src/enqueue.cc` | `DagEvCollEnqueue`, `DagEvKernelLaunch`, proxy op ID propagation |
| `src/proxy.cc` | `DagEvProxyOpBegin` / `DagEvProxyOpEnd` emission |
| `src/transport/net.cc` | Send/recv step-level events (post/xmit/done) |
| `src/transport/net_ib/p2p.cc` | `ibv_post_send`, `ibv_irecv`, `ibv_poll_cq` events |
| `tools/nccl_dag_merge.py` | Offline merge, Chrome Trace / DOT / critical-path export |

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
