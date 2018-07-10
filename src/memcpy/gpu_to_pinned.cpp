
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "memcpy/args.hpp"

static void CUDA_Memcpy_GPUToPinned(benchmark::State &state) {

  const auto flags = cudaHostAllocWriteCombined;

  state.SetLabel(fmt::format("CUDA/Memcpy/GPUToPinned/{}", "cudaHostAllocWriteCombined"));

  if (!has_cuda) {
    state.SkipWithError("CUDA/MEMCPY/GPUToPinned no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  char *src = nullptr;
  if (PRINT_IF_ERROR(cudaMalloc(&src, bytes))) {
    state.SkipWithError("CUDA/MEMCPY/GPUToPinned failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(src));

  if (PRINT_IF_ERROR(cudaMemset(src, 0, bytes))) {
    state.SkipWithError("CUDA/MEMCPY/GPUToPinned failed to perform cudaMemset");
    return;
  }

  char *dst = nullptr;
  if (PRINT_IF_ERROR(cudaHostAlloc(&dst, bytes, flags))) {
    state.SkipWithError("CUDA/MEMCPY/GPUToPinned failed to perform pinned cudaHostAlloc");
    return;
  }
  defer(cudaFreeHost(dst));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    state.PauseTiming();

    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError("CUDA/MEMCPY/GPUToPinned failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/MEMCPY/GPUToPinned failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

BENCHMARK(CUDA_Memcpy_GPUToPinned)->ALL_ARGS()->UseManualTime();
