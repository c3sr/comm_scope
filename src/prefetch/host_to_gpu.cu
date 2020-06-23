#if __CUDACC_VER_MAJOR__ >= 8

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_UM_Prefetch_HostToGPU"

auto Comm_UM_Prefetch_HostToGPU = [](benchmark::State &state,
                                     const int numa_id,
                                     const int cuda_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  if (PRINT_IF_ERROR(cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
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
    state.SkipWithError(NAME " failed to create start event");
    return;
  }
  defer(cudaEventDestroy(start));

  if (PRINT_IF_ERROR(cudaEventCreate(&stop))) {
    state.SkipWithError(NAME " failed to create end event");
    return;
  }
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId))) {
      state.SkipWithError(NAME " failed to move data to src");
      return;
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }

    cudaEventRecord(start);
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, cuda_id))) {
      state.SkipWithError(NAME " failed to move data to src");
      return;
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
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : numa::ids()) {
      std::string name = std::string(NAME)
                         + "/" + std::to_string(numa_id)
                         + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Prefetch_HostToGPU,
                                   numa_id,
                                   cuda_id)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
