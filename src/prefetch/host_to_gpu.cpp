#if CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numaum-prefetch/args.hpp"

#define NAME "NUMAUM/Prefetch/HostToGPU"

static void NUMAUM_Prefetch_HostToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));
  const int src_numa = state.range(1);
  const int dst_gpu  = state.range(2);

  numa_bind_node(src_numa);

  if (PRINT_IF_ERROR(utils::cuda_reset_device(dst_gpu))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA src device");
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
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, dst_gpu))) {
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
  state.counters.insert({{"bytes", bytes}});

  numa_bind_node(-1);
}

BENCHMARK(NUMAUM_Prefetch_HostToGPU)->Apply(ArgsCountNumaGpu)->UseManualTime();

#endif // CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1