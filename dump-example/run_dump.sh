#!/bin/bash
# =======================================================
# run_dump.sh - 单机多 rank NCCL trace（intra-host，强制走 IB）
#
# 跑一次：ncclAllReduce + ncclAllGather；导出 dump-nccl 4 层 DAG
# trace 到 ./traces/dag_<ts>/。
#
# 用法:
#   ./run_dump.sh                      # NP=4, ar=4M, ag=4M
#   ./run_dump.sh <NP>                 # 自定义 ranks (1..GPU 数量)
#   ./run_dump.sh <NP> <SIZE1> <SIZE2> # ranks + AllReduce 字节 + AllGather 字节/rank
#   SIZE 可写 4M / 64M / 1G / 1048576 等
#
# 可通过环境变量覆盖：
#   NCCL_HOME=...        指向已编译的 dump-nccl/build（默认: ../build）
#   HCA=mlx5_3           所有 rank 使用的 IB HCA（默认: mlx5_0）
#   IFNAME=ens14np0      OOB / TCP bootstrap 网卡
#   GPU_ARCH=sm_90       nvcc -gencode arch（默认 sm_90 即 H100）
#   TRACE_DIR=...        trace 输出目录
#
# 关键点：dump-nccl 仅在 host-proxy + IB verbs 路径插桩，所以脚本
# 关闭 P2P/SHM/NVLS/GIN，强制走该路径。
# =======================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMP_NCCL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NCCL_HOME="${NCCL_HOME:-${DUMP_NCCL_ROOT}/build}"
DAG_TOOL="${DUMP_NCCL_ROOT}/tools/nccl_dag_merge.py"
SRC="${SCRIPT_DIR}/minimal_ar_ag.cu"
BIN="${SCRIPT_DIR}/minimal_ar_ag"

# ---- detect MPI ----
MPI_HOME_DEFAULT="/usr/lib/x86_64-linux-gnu/openmpi"
MPI_HOME="${MPI_HOME:-${MPI_HOME_DEFAULT}}"
if [[ ! -d "${MPI_HOME}/include" ]]; then
    if command -v mpicc >/dev/null 2>&1; then
        MPI_PREFIX="$(mpicc -show 2>/dev/null | sed -nE 's|.*-I([^ ]+)/include.*|\1|p' | head -1)"
        [[ -n "$MPI_PREFIX" ]] && MPI_HOME="$MPI_PREFIX"
    fi
fi
if [[ ! -d "${MPI_HOME}/include" ]]; then
    echo "ERROR: cannot find MPI install. Set MPI_HOME=..." >&2; exit 1
fi

GPU_ARCH="${GPU_ARCH:-sm_90}"

# parse size strings like "4M", "1G", "1048576"
parse_size() {
    local s="$1"
    local num="${s//[!0-9]/}"
    local unit="${s//[0-9]/}"
    case "${unit^^}" in
        K|KB) echo $(( num * 1024 )) ;;
        M|MB) echo $(( num * 1024 * 1024 )) ;;
        G|GB) echo $(( num * 1024 * 1024 * 1024 )) ;;
        ""|B) echo "${num}" ;;
        *) echo "ERROR: bad size '$s'" >&2; exit 1 ;;
    esac
}

NP="${1:-4}"
SIZE1="${2:-4M}"     # AllReduce bytes
SIZE2="${3:-4M}"     # AllGather bytes/rank
AR_BYTES=$(parse_size "$SIZE1")
AG_BYTES=$(parse_size "$SIZE2")

# GPU 分配：每 rank 一张卡（0..NP-1）
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( NP < 1 )) || (( NP > NGPU )); then
    echo "ERROR: NP must be in 1..${NGPU}"; exit 1
fi
GPUS_ALL=()
for ((i=0; i<NGPU; i++)); do GPUS_ALL+=("$i"); done
GPUS=$(IFS=, ; echo "${GPUS_ALL[*]:0:NP}")

# 单机各 mlx5 在不同 RoCE 子网，跨 HCA QP 会 timeout，所以全部 rank 共用一个 HCA。
HCA="${HCA:-mlx5_0}"
IFNAME="${IFNAME:-ens14np0}"

TS="$(date +%Y%m%d_%H%M%S)"
TRACE_DIR="${TRACE_DIR:-${SCRIPT_DIR}/traces/dag_${TS}}"

# ---- build minimal_ar_ag if missing or stale ----
if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
    echo "[build] compiling $(basename "$SRC") (arch=${GPU_ARCH}) ..."
    arch_num="${GPU_ARCH#sm_}"
    nvcc -O2 -std=c++17 -gencode="arch=compute_${arch_num},code=${GPU_ARCH}" \
        -I "${NCCL_HOME}/include" \
        -I "${MPI_HOME}/include" \
        "$SRC" \
        -L "${NCCL_HOME}/lib" -lnccl \
        -L "${MPI_HOME}/lib" -lmpi \
        -lcudart -o "$BIN"
fi

mkdir -p "$TRACE_DIR"

echo "============================================="
echo " dump-nccl minimal trace (intra-host, IB)"
echo " host:    $(hostname -s)"
echo " ranks:   ${NP}"
echo " GPUs:    ${GPUS}"
echo " HCA:     ${HCA}"
echo " IFNAME:  ${IFNAME}"
echo " AllReduce bytes:        ${AR_BYTES} (${SIZE1})"
echo " AllGather bytes/rank:   ${AG_BYTES} (${SIZE2})"
echo " trace:   ${TRACE_DIR}"
echo "============================================="

mpirun -np "${NP}" --allow-run-as-root --bind-to none \
    --host "$(hostname -s):${NP}" \
    --mca btl_tcp_if_include "${IFNAME}" \
    --mca oob_tcp_if_include "${IFNAME}" \
    -x LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}" \
    -x NCCL_DAG_TRACE=1 -x NCCL_DAG_TRACE_DIR="${TRACE_DIR}" \
    -x NCCL_GIN_ENABLE=0 -x NCCL_P2P_DISABLE=1 -x NCCL_SHM_DISABLE=1 -x NCCL_NVLS_ENABLE=0 \
    -x NCCL_IB_DISABLE=0 -x NCCL_NET=IB \
    -x NCCL_NET_GDR_LEVEL=PHB -x NCCL_IB_TC=96 \
    -x NCCL_SOCKET_IFNAME="${IFNAME}" \
    -x NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -x GPUS="${GPUS}" -x HCA_ENV="${HCA}" \
    bash -c '
        L=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
        IFS=, read -ra GA <<< "$GPUS"
        export CUDA_VISIBLE_DEVICES="${GA[$L]}"
        export NCCL_IB_HCA="$HCA_ENV"
        exec "$@"
    ' -- "$BIN" "$AR_BYTES" "$AG_BYTES"

echo
echo "----- traces -----"
ls -la "${TRACE_DIR}"/nccl_dag_*.jsonl 2>/dev/null || { echo "no traces"; exit 1; }

if command -v python3 >/dev/null && [[ -f "${DAG_TOOL}" ]]; then
    python3 "${DAG_TOOL}" "${TRACE_DIR}"/nccl_dag_*.jsonl --summary || true
    python3 "${DAG_TOOL}" "${TRACE_DIR}"/nccl_dag_*.jsonl -o "${TRACE_DIR}/merged.json" >/dev/null
    python3 "${DAG_TOOL}" "${TRACE_DIR}"/nccl_dag_*.jsonl --chrome-trace "${TRACE_DIR}/chrome_trace.json" >/dev/null
    echo
    echo "Outputs:"
    ls "${TRACE_DIR}"
    echo
    echo "open chrome://tracing or Perfetto and load:"
    echo "  ${TRACE_DIR}/chrome_trace.json"
fi
