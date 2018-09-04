#if CUDA_VERSION_MAJOR >= 8

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#endif

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm/UM/Latency/HostToGPU"

template <bool NOOP = false>
__global__ void gpu_traverse(size_t *ptr, const size_t steps) {

  if (NOOP) {
    return;
  }
  size_t next = 0;
  for (int i = 0; i < steps; ++i) {
    next = ptr[next];
  }
  ptr[next] = 1;
}

static void Comm_UM_Latency_HostToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t steps = state.range(0);
  #if USE_NUMA
  const int numa_id  = state.range(1);
  const int cuda_id  = state.range(2);
  #else
  const int cuda_id = state.range(1);
  #endif

  const size_t stride = 65536 * 2;
  const size_t bytes  = sizeof(size_t) * (steps + 1) * stride;

#if USE_NUMA
  numa_bind_node(numa_id);
#endif

  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA dst device");
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
      state.SkipWithError(NAME " failed to prefetch to CPU");
      return;
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }

    cudaEventRecord(start);
    gpu_traverse<<<1, 1>>>(ptr, steps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millis = 0;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
  }
  state.counters["strides"] = steps;

#if USE_NUMA
  numa_bind_node(-1);
#endif
}

#if USE_NUMA
BENCHMARK(Comm_UM_Latency_HostToGPU)->Apply(ArgsCountNumaGpu)->UseManualTime();
#else
BENCHMARK(Comm_UM_Latency_HostToGPU)->Apply(ArgsCountGpu)->UseManualTime();
#endif

#endif // CUDA_VERSION_MAJOR >= 8 
