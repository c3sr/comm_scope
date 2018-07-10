#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "memcpy/args.hpp"

static void CUDA_Memcpy_HostToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError("CUDA/MEMCPY/HostToGPU no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  char *src        = new char[bytes];
  char *dst        = nullptr;

  defer(delete[] src);

  if (PRINT_IF_ERROR(cudaMalloc(&dst, bytes))) {
    state.SkipWithError("CUDA/MEMCPY/HostToGPU failed to perform cudaMemcpy");
    return;
  }
  defer(cudaFree(dst));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    state.PauseTiming();

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError("CUDA/MEMCPY/HostToGPU failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/MEMCPY/HostToGPU failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

BENCHMARK(CUDA_Memcpy_HostToGPU)->ALL_ARGS()->UseManualTime();
