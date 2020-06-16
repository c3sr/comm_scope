#if __CUDACC_VER_MAJOR__ >= 8

 #include "sysbench/sysbench.hpp"
 
#include "args.hpp"

#define NAME "Comm_UM_Prefetch_GPUToGPU"

auto Comm_UM_Prefetch_GPUToGPU = [](benchmark::State &state, const int src_gpu, const int dst_gpu) {

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));

  if (PRINT_IF_ERROR(cuda_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA src device");
    return;
  }
  if (PRINT_IF_ERROR(cuda_reset_device(dst_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA src device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA dst device");
    return;
  }

  char *ptr = nullptr;
  if (PRINT_IF_ERROR(cudaMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMallocManaged");
    return;
  }
  defer(cudaFree(ptr));

  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }

  cudaEvent_t start, stop;
  if (PRINT_IF_ERROR(cudaEventCreate(&start))) {
    state.SkipWithError(NAME " failed to create event");
    return;
  }
  defer(cudaEventDestroy(start));

  if (PRINT_IF_ERROR(cudaEventCreate(&stop))) {
    state.SkipWithError(NAME " failed to create event");
    return;
  }
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {
    cudaMemPrefetchAsync(ptr, bytes, src_gpu);
    cudaSetDevice(src_gpu);
    cudaDeviceSynchronize();
    cudaSetDevice(dst_gpu);
    cudaDeviceSynchronize();
    if (PRINT_IF_ERROR(cudaGetLastError())) {
      state.SkipWithError(NAME " failed to prep iteration");
      return;
    }

    cudaEventRecord(start);
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, dst_gpu))) {
      state.SkipWithError(NAME " failed prefetch");
      break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millis = 0;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
};

static void registerer() {
  for (size_t i = 0; i <  unique_cuda_device_ids().size(); ++i) {
    for (size_t j = i + 1; j < unique_cuda_device_ids().size(); ++j) {
      auto src_gpu = unique_cuda_device_ids()[i];
      auto dst_gpu = unique_cuda_device_ids()[j];
      std::string name = std::string(NAME) + "/" + std::to_string(src_gpu) + "/" + std::to_string(dst_gpu);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Prefetch_GPUToGPU, src_gpu, dst_gpu)->SMALL_ARGS()->UseManualTime();
    }
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
