# dump-example — minimal NCCL DAG trace

一个最小可复现样例：用插桩过的 `dump-nccl` 跑一次 `ncclAllReduce` +
一次 `ncclAllGather`，导出 4 层（api / kernel / proxy / network）DAG trace，
并合并成 `merged.json` + `chrome_trace.json`。

```
dump-example/
├── README.md            ← 本文件
├── minimal_ar_ag.cu     ← MPI bootstrap + AllReduce + AllGather
├── run_dump.sh          ← 一键编译 + mpirun + 合并 trace
└── traces/              ← (运行后生成) dag_<ts>/...
```

---

## 1. 前置条件

| 组件 | 说明 |
|---|---|
| Linux + 一台/多卡 NVIDIA GPU | 默认假设 H100 (`sm_90`)；其他卡用 `GPU_ARCH=sm_80` 等覆盖 |
| CUDA Toolkit | 提供 `nvcc`、`libcudart`，建议 12.x+ |
| OpenMPI | 提供 `mpirun`、`libmpi`；脚本会自动探测 `mpicc -show` |
| RDMA stack | `rdma-core` / `libibverbs` / `ibverbs-providers`；至少一个 `mlx5_*` HCA up |
| Python 3 | 用来跑 `nccl_dag_merge.py`（`--summary` / `--chrome-trace`） |
| dump-nccl 已编译 | 即 `dump-nccl/build/{lib/libnccl.so.2, include/nccl.h}` 存在 |

确认一下：

```bash
nvidia-smi -L                 # 列出 GPU
ibv_devices                   # 列出 RDMA 设备，至少有 mlx5_0
mpirun --version              # OpenMPI ≥ 4.x
nvcc --version                # CUDA toolchain
ls ../build/lib/libnccl.so.2  # dump-nccl 已编译
```

如果还没编译 dump-nccl：

```bash
cd ..                          # 进入 dump-nccl/
make -j src.build              # 产出 build/lib/libnccl.so.2
```

---

## 2. 一键运行

```bash
cd dump-nccl/dump-example
./run_dump.sh                  # NP=4, AllReduce=4M, AllGather=4M/rank
./run_dump.sh 4 4M 8M          # 自定义 size
./run_dump.sh 2 64M 64M        # 2 rank, 64 MiB
HCA=mlx5_3 ./run_dump.sh 4     # 切换 HCA
GPU_ARCH=sm_80 ./run_dump.sh   # A100
```

输出：

```
traces/dag_<ts>/
├── nccl_dag_<pid>.jsonl   × NP        # 每个 rank 一份原始事件流
├── merged.json                        # 合并 + 跨 rank 边
└── chrome_trace.json                  # 拖进 chrome://tracing 或 ui.perfetto.dev
```

期望看到的 `--summary`：

```
Nodes by layer:
  api     : 8
  kernel  : 8
  proxy   : 2144
  network : 1168
Cross-rank edges: 288
```

---

## 3. 在新环境下从零配置（Ubuntu 22.04 示例）

```bash
# 1. 系统包
sudo apt update
sudo apt install -y build-essential git python3 \
                    libopenmpi-dev openmpi-bin \
                    rdma-core libibverbs-dev ibverbs-providers

# 2. CUDA toolkit（按 NVIDIA 官网说明装；下面以 12.4 为例）
#   确保 nvcc 在 PATH，libcudart 在 LD_LIBRARY_PATH
nvcc --version

# 3. 拉 / 准备 dump-nccl 源码并编译（<repo> 是 dump-nccl 根目录）
cd <repo>
make -j src.build
# 产物: build/lib/libnccl.so.2 与 build/include/nccl.h

# 4. 跑样例
cd dump-example
./run_dump.sh 4 4M 8M
```

---

## 4. 必须的 NCCL 环境变量（脚本已自动设置）

| 变量 | 作用 |
|---|---|
| `NCCL_DAG_TRACE=1` | 打开 dump-nccl tracer |
| `NCCL_DAG_TRACE_DIR=<dir>` | 每个进程写 `nccl_dag_<pid>.jsonl` 到该目录 |
| `NCCL_GIN_ENABLE=0` | 禁用 device-initiated GIN/RMA 路径（未插桩） |
| `NCCL_P2P_DISABLE=1` `NCCL_SHM_DISABLE=1` `NCCL_NVLS_ENABLE=0` | 禁用 NVLink/SHM/NVLS，强制走 IB |
| `NCCL_NET=IB` `NCCL_IB_DISABLE=0` | 强制 IB verbs 网络后端 |
| `NCCL_NET_GDR_LEVEL=PHB` `NCCL_IB_TC=96` | 启用 GDR + 设 ToS |
| `NCCL_IB_HCA=$HCA` | 单机模式所有 rank 同一 HCA（避免跨 RoCE 子网） |
| `NCCL_SOCKET_IFNAME=$IFNAME` | OOB/bootstrap 网卡 |
| `LD_LIBRARY_PATH=<dump-nccl>/build/lib:...` | 优先加载插桩 NCCL，而不是系统 NCCL |

---

## 5. 常见问题

**Q: trace 里只有 `coll_enqueue`，没有 kernel/proxy/network 事件**
- 检查 `NP > 1`（`nranks=1` 时走 `ncclLaunchOneRank`，不会下发到 kernel/proxy/network）
- 检查 `LD_LIBRARY_PATH` 是否真的加载了 `dump-nccl/build/lib/libnccl.so.2`，
  可在 `mpirun` 里加 `-x NCCL_DEBUG=VERSION` 观察
- 检查 `NCCL_GIN_ENABLE=0`、`NCCL_P2P_DISABLE=1` 等是否生效

**Q: `ibv_modify_qp ... Connection timed out`，跨 GID/子网**
- 单机 8 个 mlx5 通常各处独立 RoCE 子网；脚本默认让所有 rank 用同一个 `mlx5_0`，可用 `HCA=mlx5_3 ./run_dump.sh ...` 切换

**Q: `nvcc fatal: Unsupported gpu architecture 'compute_90'`**
- 你的 GPU 不是 H100。用 `GPU_ARCH=sm_80` (A100) / `sm_86` (A40/L40) / `sm_70` (V100) 等覆盖

**Q: 找不到 `mpi.h` / `libmpi.so`**
- 设置 `MPI_HOME=/path/to/openmpi`，要求其下有 `include/mpi.h` 和 `lib/libmpi.so`

---

## 6. 想拿哪些信息？

- `merged.json`：所有 rank 节点 + 跨 rank send→recv 边，可后处理（关键路径、瓶颈）
- `chrome_trace.json`：直接打开 `chrome://tracing` 或 https://ui.perfetto.dev 看时序
- `--summary`：分层事件计数 / 总字节 / 时间跨度
- 还可以用 `tools/nccl_dag_merge.py --critical-path` / `--dot` 等
