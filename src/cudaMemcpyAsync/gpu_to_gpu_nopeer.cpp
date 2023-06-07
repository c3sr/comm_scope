#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyAsync_GPUToGPU"

auto Comm_cudaMemcpyAsync_GPUToGPU = [](benchmark::State &state,
                                        const int numa_id, const int src_gpu,
                                        const int dst_gpu) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  char *src = nullptr;
  char *dst = nullptr;

  OR_SKIP_AND_RETURN(scope::cuda_reset_device(src_gpu),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(scope::cuda_reset_device(dst_gpu),
                     NAME " failed to reset CUDA device");
  OR_SKIP_AND_RETURN(cudaSetDevice(src_gpu), NAME " failed to set src device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src, bytes),
                     NAME " failed to perform cudaMalloc");
  defer(cudaFree(src));
  OR_SKIP_AND_RETURN(cudaMemset(src, 0, bytes),
                     NAME " failed to perform src cudaMemset");

  cudaError_t err = cudaDeviceDisablePeerAccess(dst_gpu);
  if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
    state.SkipWithError(NAME " failed to disable peer access");
    return;
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(dst_gpu), NAME " failed to set dst device");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, bytes),
                     NAME " failed to perform cudaMalloc");
  defer(cudaFree(dst));
  OR_SKIP_AND_RETURN(cudaMemset(dst, 0, bytes),
                     NAME " failed to perform dst cudaMemset");

  err = cudaDeviceDisablePeerAccess(src_gpu);
  if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
    state.SkipWithError(NAME " failed to disable peer access");
    return;
  }

  cudaEvent_t start, stop;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "failed to create start event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "failed to create stop event");

  for (auto _ : state) {

    OR_SKIP_AND_BREAK(cudaEventRecord(start, NULL), "failed to record start");
    OR_SKIP_AND_BREAK(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice),
        "failed to cudaMemcpyAsync");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, NULL), "failed to record start");
    ;
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "failed to sync");
    ;

    float msecTotal = 0.0f;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&msecTotal, start, stop),
                      "failed to get elapsed time");
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
  state.counters["numa_id"] = numa_id;

  // re-enable peer access
  err = cudaSetDevice(src_gpu);
  err = cudaDeviceEnablePeerAccess(dst_gpu, 0);
  // docs say should return cudaErrorInvalidDevice, actually returns
  // cudaErrorPeerAccessUnsupported?
  if (cudaSuccess != err && cudaErrorInvalidDevice != err &&
      cudaErrorPeerAccessUnsupported != err) {
    PRINT_IF_ERROR(err);
    state.SkipWithError(NAME " couldn't re-enable peer access");
  }
  err = cudaSetDevice(dst_gpu);
  err = cudaDeviceEnablePeerAccess(src_gpu, 0);
  if (cudaSuccess != err && cudaErrorInvalidDevice != err &&
      cudaErrorPeerAccessUnsupported != err) {
    PRINT_IF_ERROR(err);
    state.SkipWithError(NAME " couldn't re-enable peer access");
  }
};

static void registerer() {

  const std::vector<MemorySpace> cudaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::cuda_device);
  const std::vector<MemorySpace> numaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (const auto &src : cudaSpaces) {
    for (const auto &dst : cudaSpaces) {
      auto src_gpu = src.device_id();
      auto dst_gpu = dst.device_id();
      for (const auto &numa : numaSpaces) {
        const auto numa_id = numa.numa_id();
        const std::string name =
            std::string(NAME) + "/" + std::to_string(numa_id) + "/" +
            std::to_string(src_gpu) + "/" + std::to_string(dst_gpu);
        benchmark::RegisterBenchmark(name.c_str(),
                                     Comm_cudaMemcpyAsync_GPUToGPU, numa_id,
                                     src_gpu, dst_gpu)
            ->SMALL_ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
