#include <cassert>
#include <cuda_runtime.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm_MemcpyPeer"

#define OR_SKIP(stmt, msg)                                                                                             \
  if (PRINT_IF_ERROR(stmt)) {                                                                                          \
    state.SkipWithError(msg);                                                                                          \
    return;                                                                                                            \
  }

auto Comm_MemcpyPeer = [](benchmark::State &state, const int srcGpu, const int dstGpu) {
  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP(utils::cuda_reset_device(srcGpu), NAME " failed to reset src CUDA device");
  OR_SKIP(utils::cuda_reset_device(dstGpu), NAME " failed to reset dst CUDA device");

  void *src = nullptr;
  void *dst = nullptr;
  cudaStream_t stream;
  cudaError_t err;
  cudaEvent_t start, stop;

  OR_SKIP(cudaSetDevice(srcGpu), NAME " failed to set src device");
  OR_SKIP(cudaMalloc(&src, bytes), NAME " failed to perform cudaMalloc");
  defer(cudaFree(src));
  OR_SKIP(cudaMemset(src, 0, bytes), NAME " failed to perform src cudaMemset");
  OR_SKIP(cudaStreamCreate(&stream), NAME " failed to create stream");
  OR_SKIP(cudaEventCreate(&start), NAME " couldn't create start event");
  OR_SKIP(cudaEventCreate(&stop), NAME " couldn't create stop event");
  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));
  if (srcGpu != dstGpu) {
    err = cudaDeviceDisablePeerAccess(dstGpu);
    if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
      state.SkipWithError(NAME " failed to disable peer access");
      return;
    }
  }

  OR_SKIP(cudaSetDevice(dstGpu), NAME " failed to set dst device");
  OR_SKIP(cudaMalloc(&dst, bytes), NAME " failed to perform cudaMalloc");
  defer(cudaFree(dst));
  OR_SKIP(cudaMemset(dst, 0, bytes), NAME " failed to perform dst cudaMemset");
  if (srcGpu != dstGpu) {
    err = cudaDeviceDisablePeerAccess(srcGpu);
    if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
      state.SkipWithError(NAME " failed to disable peer access");
      return;
    }
  }

  OR_SKIP(cudaSetDevice(srcGpu), NAME " failed to set src device");
  for (auto _ : state) {
    OR_SKIP(cudaEventRecord(start, stream), NAME " failed to record start");
    OR_SKIP(cudaMemcpyPeerAsync(dst, dstGpu, src, srcGpu, bytes, stream), NAME " failed to memcpy");
    OR_SKIP(cudaEventRecord(stop, stream), NAME " failed to stop");
    OR_SKIP(cudaEventSynchronize(stop), NAME " failed to synchronize");

    float ms = 0.0f;
    OR_SKIP(cudaEventElapsedTime(&ms, start, stop), NAME "failed to compute elapsed time");
    state.SetIterationTime(ms / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"]  = bytes;
  state.counters["srcGpu"] = srcGpu;
  state.counters["dstGpu"] = dstGpu;
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i; j < unique_cuda_device_ids().size(); ++j) {
      auto srcGpu = unique_cuda_device_ids()[i];
      auto dstGpu = unique_cuda_device_ids()[j];
      name        = std::string(NAME) + "/" + std::to_string(srcGpu) + "/" + std::to_string(dstGpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_MemcpyPeer, srcGpu, dstGpu)->SMALL_ARGS()->UseManualTime();
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer, NAME);
