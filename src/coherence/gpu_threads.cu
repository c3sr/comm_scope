#if CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"

#define NAME "Comm/UM/Coherence/GPUThreads"

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

template <bool NOOP = false>
__global__ void gpu_write2(char *ptr, const size_t count, const size_t stride) {
  if (NOOP) {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  const size_t wx       = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  // split bytes into numWarps chunks
  const size_t bi = wx * (count / numWarps);
  const size_t ei = (wx + 1) * (count / numWarps);

  if (0 == lx) {
    for (size_t i = bi; i < ei && i < count; i += stride) {
      ptr[i] = 0;
    }
  }
}

static void Comm_UM_Coherence_GPUThreads(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const size_t pageSize = page_size();
  const size_t stride   = pageSize;

  const auto warps     = 1ULL << static_cast<size_t>(state.range(0));
  const size_t threads = warps * 32;
  const int src_numa   = state.range(1);
  const int dst_gpu    = state.range(2);
  const size_t bytes   = 1ULL << 28;

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

  dim3 blockDim(min(threads, size_t(128)));
  dim3 gridDim((threads + blockDim.x - 1) / blockDim.x);

  for (auto _ : state) {
    state.PauseTiming();

    // if (threads * stride > bytes) {
    //   state.SkipWithError("ah");
    // }

    cudaError_t err = cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId);
    if (err == cudaErrorInvalidDevice) {
      for (size_t i = 0; i < bytes; i += pageSize) {
        ptr[i] = 0;
      }
    }

    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }

    state.ResumeTiming();
    cudaEventRecord(start);
    gpu_write2<<<gridDim, blockDim>>>(ptr, bytes, stride);
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

// BENCHMARK(Comm_UM_Coherence_GPUThreads)->Apply(ArgsThreadsNumaGpu)->UseManualTime();

#endif // CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1