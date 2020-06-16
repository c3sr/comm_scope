#include "sysbench/sysbench.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyAsync_GPUToGPUPeer"

auto Comm_Memcpy_GPUToGPUPeer = [](benchmark::State &state, const int src_gpu,
                                   const int dst_gpu) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP(cuda_reset_device(src_gpu), NAME " failed to reset src CUDA device");
  OR_SKIP(cuda_reset_device(dst_gpu), NAME " failed to reset dst CUDA device");

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
    OR_SKIP(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice),
            NAME " failed to memcpy");
    OR_SKIP(cudaEventRecord(stop, NULL), NAME " failed to stop");
    OR_SKIP(cudaEventSynchronize(stop), NAME " failed to synchronize");

    float msecTotal = 0.0f;
    OR_SKIP(cudaEventElapsedTime(&msecTotal, start, stop),
            NAME "failed to compute elapsed time");
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i + 1; j < unique_cuda_device_ids().size(); ++j) {
      auto src_gpu = unique_cuda_device_ids()[i];
      auto dst_gpu = unique_cuda_device_ids()[j];
      int s2d, d2s;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&s2d, src_gpu, dst_gpu)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&d2s, dst_gpu, src_gpu))) {
        if (s2d && d2s) {
          name = std::string(NAME) + "/" + std::to_string(src_gpu) + "/" +
                 std::to_string(dst_gpu);
          benchmark::RegisterBenchmark(name.c_str(), Comm_Memcpy_GPUToGPUPeer,
                                       src_gpu, dst_gpu)
              ->SMALL_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);
