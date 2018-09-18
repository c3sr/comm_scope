#if CUDA_VERSION_MAJOR >= 8

#include <cassert>

#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm/UM/Coherence/GPUToGPU"

template <bool NOOP = false>
__global__ void gpu_write(char *ptr, const size_t count, const size_t stride) {
  if (NOOP) {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx             = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

static void Comm_UM_Coherence_GPUToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));

  if (num_gpus() < 2) {
    state.SkipWithError(NAME "requires at least 2 GPUs");
    return;
  }
  const int src_gpu = FLAG(cuda_device_ids)[0];
  const int dst_gpu = FLAG(cuda_device_ids)[1];

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
    gpu_write<<<256, 256>>>(ptr, bytes, pageSize);
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
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
}

BENCHMARK(Comm_UM_Coherence_GPUToGPU)->SMALL_ARGS()->UseManualTime();

#endif // CUDA_VERSION_MAJOR >= 8
