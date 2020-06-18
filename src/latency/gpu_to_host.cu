#if __CUDACC_VER_MAJOR__ >= 8

 #include "sysbench/sysbench.hpp"

#include "args.hpp"

#define NAME "Comm_UM_Latency_GPUToHost"

template <bool NOOP = false>
void cpu_traverse(size_t *ptr, const size_t steps) {

  if (NOOP) {
    return;
  }
  size_t next = 0;
  for (size_t i = 0; i < steps; ++i) {
    next = ptr[next];
  }
  ptr[next] = 1;
}

auto Comm_UM_Latency_GPUToHost = [] (benchmark::State &state,
  const int numa_id,
  const int cuda_id) {

  const size_t steps = state.range(0);

  const size_t stride = 65536 * 2;
  const size_t bytes  = sizeof(size_t) * (steps + 1) * stride;
  numa::ScopedBind binder(numa_id);
  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(cudaDeviceReset())) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  size_t *ptr = nullptr;
  if (PRINT_IF_ERROR(cudaMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMallocManaged");
    return;
  }
  defer(cudaFree(ptr));

  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }

  // set up stride pattern
  for (size_t i = 0; i < steps; ++i) {
    ptr[i * stride] = (i + 1) * stride;
  }
  if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
    state.SkipWithError(NAME " failed to synchronize");
    return;
  }

  for (auto _ : state) {
    state.PauseTiming();
    // prefetch to source
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, cuda_id))) {
      state.SkipWithError(NAME " failed prefetch to device");
      return;
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }
    state.ResumeTiming();

    cpu_traverse(ptr, steps);
  }
  state.counters["strides"] = steps;
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : numa::ids()) {
      std::string name = std::string(NAME)
                       + "/" + std::to_string(numa_id) 
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Latency_GPUToHost,
        numa_id,
        cuda_id)->SMALL_ARGS();
    }
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8