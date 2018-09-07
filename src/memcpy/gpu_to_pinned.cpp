#include <cassert>
#include <cuda_runtime.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "CUDA/Memcpy/GpuToPinned"

#define OR_SKIP(stmt, msg) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(msg); \
    return; \
  }

static void CUDA_Memcpy_GPUToPinned(benchmark::State &state) {

  const auto flags = cudaHostAllocWriteCombined;

  state.SetLabel(fmt::format("CUDA/Memcpy/GPUToPinned/{}", "cudaHostAllocWriteCombined"));

  if (!has_cuda) {
    state.SkipWithError("CUDA/MEMCPY/GPUToPinned no CUDA device found");
    return;
  }

  const int cuda_id = FLAG(cuda_device_ids)[0];
  OR_SKIP(cudaSetDevice(cuda_id), NAME " failed to set CUDA device");

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
  state.counters["cuda_id"] = cuda_id;
}

BENCHMARK(CUDA_Memcpy_GPUToPinned)->SMALL_ARGS()->UseManualTime();
