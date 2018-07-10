#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "memcpy/args.hpp"

static void CUDA_Memcpy_GPUToHost(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError("CUDA/MEMCPY/GPUToHost no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  char *src        = nullptr;
  char *dst        = new char[bytes];

  defer(delete[] dst);

  if (PRINT_IF_ERROR(cudaMalloc(&src, bytes))) {
    state.SkipWithError("CUDA/MEMCPY/GPUToHost failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(src));

  if (PRINT_IF_ERROR(cudaMemset(src, 0, bytes))) {
    state.SkipWithError("CUDA/MEMCPY/GPUToHost failed to perform cudaMemset");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    state.PauseTiming();

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError("CUDA/MEMCPY/GPUToHost failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/MEMCPY/GPUToHost failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

BENCHMARK(CUDA_Memcpy_GPUToHost)->ALL_ARGS()->UseManualTime();
