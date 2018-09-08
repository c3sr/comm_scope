#include <cassert>

#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm/Memcpy/GPUToGPUPeer"

#define OR_SKIP(stmt, msg) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(msg); \
    return; \
  }

static void CUDA_Memcpy_GPUToGPUPeer(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (num_gpus() < 2) {
    state.SkipWithError(NAME " requires >1 CUDA GPUs");
    return;
  }

  assert(FLAG(cuda_device_ids).size() >= 2);
  const int src_gpu = FLAG(cuda_device_ids)[0];
  const int dst_gpu = FLAG(cuda_device_ids)[1];

  if (src_gpu == dst_gpu) {
    state.SkipWithError(NAME " requires two different GPUs");
    return;
  }

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP(utils::cuda_reset_device(src_gpu), NAME " failed to reset src CUDA device");
  OR_SKIP(utils::cuda_reset_device(dst_gpu), NAME " failed to reset dst CUDA device");

  char *src = nullptr;
  char *dst = nullptr;

  OR_SKIP(cudaSetDevice(src_gpu), NAME " failed to set src device");
  OR_SKIP(cudaMalloc(&src, bytes), NAME " failed to perform cudaMalloc");
  defer(cudaFree(src));
  OR_SKIP(cudaMemset(src, 0, bytes), NAME " failed to perform src cudaMemset");
  cudaError_t err = cudaDeviceEnablePeerAccess(dst_gpu, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError(NAME " failed to ensure peer access");
    return;
  }

  OR_SKIP(cudaSetDevice(dst_gpu), NAME " failed to set dst device");
  OR_SKIP(cudaMalloc(&dst, bytes), NAME " failed to perform cudaMalloc");
  defer(cudaFree(dst));
  OR_SKIP(cudaMemset(dst, 0, bytes), NAME " failed to perform dst cudaMemset");
  err = cudaDeviceEnablePeerAccess(src_gpu, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError(NAME " failed to ensure peer access");
    return;
  }

  cudaEvent_t start, stop;
  OR_SKIP(cudaEventCreate(&stop), NAME " couldn't create stop event");
  OR_SKIP(cudaEventCreate(&start), NAME " couldn't create start event");

  for (auto _ : state) {
    OR_SKIP(cudaEventRecord(start, NULL), NAME " failed to record start");
    OR_SKIP(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice), NAME " failed to memcpy");
    OR_SKIP(cudaEventRecord(stop, NULL), NAME " failed to stop");
    OR_SKIP(cudaEventSynchronize(stop), NAME " failed to synchronize");

    float msecTotal = 0.0f;
   OR_SKIP(cudaEventElapsedTime(&msecTotal, start, stop), NAME "failed to compute elapsed time");
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
}

BENCHMARK(CUDA_Memcpy_GPUToGPUPeer)->SMALL_ARGS()->UseManualTime();
