#if CUDA_VERSION_MAJOR >= 8

#include <cassert>

#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#include "init/numa.hpp"
#endif // USE_NUMA

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"

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
#if USE_NUMA
  const int numa_id,
#endif // USE_NUMA
  const int cuda_id) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t steps = state.range(0);

  const size_t stride = 65536 * 2;
  const size_t bytes  = sizeof(size_t) * (steps + 1) * stride;
#if USE_NUMA
  numa_bind_node(numa_id);
#endif
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
#if USE_NUMA
  state.counters["numa_id"] = numa_id;
#endif

#if USE_NUMA
  // reset to run on any node
  numa_bind_node(-1);
#endif
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
#if USE_NUMA
    for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Latency_GPUToHost,
#if USE_NUMA
        numa_id,
#endif // USE_NUMA
        cuda_id)->SMALL_ARGS();
#if USE_NUMA
    }
#endif // USE_NUMA
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // CUDA_VERSION_MAJOR >= 8