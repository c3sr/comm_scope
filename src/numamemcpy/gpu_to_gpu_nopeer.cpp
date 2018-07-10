#if USE_NUMA == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "numamemcpy/args.hpp"
#include "utils/utils.hpp"

#define NAME "NUMA/Memcpy/GPUToGPU"

static void NUMA_Memcpy_GPUToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " no NUMA control available");
    return;
  }

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));
  const int numa_id = state.range(1);
  const int src_gpu = state.range(2);
  const int dst_gpu = state.range(3);

  numa_bind_node(numa_id);

  char *src = nullptr;
  char *dst = nullptr;

  if (PRINT_IF_ERROR(utils::cuda_reset_device(src_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(dst_gpu))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(src_gpu))) {
    state.SkipWithError(NAME " failed to set src device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&src, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(src));
  if (PRINT_IF_ERROR(cudaMemset(src, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform src cudaMemset");
    return;
  }
  cudaError_t err = cudaDeviceDisablePeerAccess(dst_gpu);
  if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
    state.SkipWithError(NAME " failed to disable peer access");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set dst device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&dst, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(dst));
  if (PRINT_IF_ERROR(cudaMemset(dst, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform dst cudaMemset");
    return;
  }
  err = cudaDeviceDisablePeerAccess(src_gpu);
  if (cudaSuccess != err && cudaErrorPeerAccessNotEnabled != err) {
    state.SkipWithError(NAME " failed to disable peer access");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {

    cudaEventRecord(start, NULL);
    const auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError(NAME " failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  // re-enable NUMA scheduling
  numa_bind_node(-1);

  // re-enable peer access
  err = cudaSetDevice(src_gpu);
  err = cudaDeviceEnablePeerAccess(dst_gpu, 0);
  // docs say should return cudaErrorInvalidDevice, actually returns cudaErrorPeerAccessUnsupported?
  if (cudaSuccess != err && cudaErrorInvalidDevice != err && cudaErrorPeerAccessUnsupported != err) {
    PRINT_IF_ERROR(err);
    state.SkipWithError(NAME " couldn't re-enable peer access");
  }
  err = cudaSetDevice(dst_gpu);
  err = cudaDeviceEnablePeerAccess(src_gpu, 0);
  if (cudaSuccess != err && cudaErrorInvalidDevice != err && cudaErrorPeerAccessUnsupported != err) {
    PRINT_IF_ERROR(err);
    state.SkipWithError(NAME " couldn't re-enable peer access");
  }

  numa_bind_node(-1);
}

BENCHMARK(NUMA_Memcpy_GPUToGPU)->Apply(ArgsCountNumaGpuGpuNoSelf)->UseManualTime();

#endif // USE_NUMA == 1