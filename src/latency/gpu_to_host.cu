#if CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numaum-latency/args.hpp"

#define NAME "NUMAUM/Latency/GPUToHost"

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

static void NUMAUM_Latency_GPUToHost(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const size_t steps = state.range(0);
  const int numa_id  = state.range(1);
  const int cuda_id  = state.range(2);

  const size_t stride = 65536 * 2;
  const size_t bytes  = sizeof(size_t) * (steps + 1) * stride;

  numa_bind_node(numa_id);
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
      state.SkipWithError(NAME " failed to synchronize");
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

  // reset to run on any node
  numa_bind_node(-1);
}

BENCHMARK(NUMAUM_Latency_GPUToHost)->Apply(ArgsCountNumaGpu)->MinTime(0.1);

#endif // CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1