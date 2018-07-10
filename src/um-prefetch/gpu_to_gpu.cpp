#if CUDA_VERSION_MAJOR >= 8

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "um-prefetch/args.hpp"

#define NAME "UM/Prefetch/GPUToGPU"

static void UM_Prefetch_GPUToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));
  const int src_gpu = state.range(1);
  const int dst_gpu = state.range(2);

  if (PRINT_IF_ERROR(utils::cuda_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA src device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(dst_gpu))) {
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
      state.SkipWithError("CUDA/MEMCPY/HostToGPU failed prefetch");
      break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millis = 0;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError("CUDA/MEMCPY/HostToGPU failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

BENCHMARK(UM_Prefetch_GPUToGPU)->Apply(ArgsCountGpuGpuNoSelf)->MinTime(0.1)->UseManualTime();

#endif // CUDA_VERSION_MAJOR >= 8