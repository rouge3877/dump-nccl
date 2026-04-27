// minimal_ar_ag.cu
//
// 最小 NCCL 程序：一次 ncclAllReduce + 一次 ncclAllGather。
// 用法: minimal_ar_ag <ar_bytes> <ag_bytes_per_rank>
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define NCCLCHECK(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { \
    fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size_t ar_bytes = (argc > 1) ? strtoull(argv[1], nullptr, 0) : (4ull << 20);
  size_t ag_bytes = (argc > 2) ? strtoull(argv[2], nullptr, 0) : (4ull << 20);
  size_t ar_count = ar_bytes / sizeof(float); if (ar_count == 0) ar_count = 1;
  size_t ag_count = ag_bytes / sizeof(float); if (ag_count == 0) ag_count = 1;

  CUDACHECK(cudaSetDevice(0));

  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  float *d_ar_send = nullptr, *d_ar_recv = nullptr;
  float *d_ag_send = nullptr, *d_ag_recv = nullptr;
  CUDACHECK(cudaMalloc(&d_ar_send, ar_count * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_ar_recv, ar_count * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_ag_send, ag_count * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_ag_recv, ag_count * (size_t)nranks * sizeof(float)));
  CUDACHECK(cudaMemset(d_ar_send, rank + 1, ar_count * sizeof(float)));
  CUDACHECK(cudaMemset(d_ag_send, rank + 1, ag_count * sizeof(float)));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  NCCLCHECK(ncclAllReduce(d_ar_send, d_ar_recv, ar_count, ncclFloat, ncclSum, comm, stream));
  NCCLCHECK(ncclAllGather(d_ag_send, d_ag_recv, ag_count, ncclFloat, comm, stream));

  CUDACHECK(cudaStreamSynchronize(stream));

  if (rank == 0) {
    printf("[rank 0] nranks=%d  AllReduce=%zu B  AllGather=%zu B/rank\n",
           nranks, ar_count * sizeof(float), ag_count * sizeof(float));
  }

  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(d_ar_send));
  CUDACHECK(cudaFree(d_ar_recv));
  CUDACHECK(cudaFree(d_ag_send));
  CUDACHECK(cudaFree(d_ag_recv));
  NCCLCHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return 0;
}
