#if CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "numaum-coherence/args.hpp"
#include "utils/utils.hpp"

#define NAME "NUMAUM/Coherence/HostToGPU"

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

static void NUMAUM_Coherence_HostToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const size_t pageSize = page_size();

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
  state.counters.insert({{"bytes", bytes}});

  numa_bind_node(-1);
}

BENCHMARK(NUMAUM_Coherence_HostToGPU)->Apply(ArgsCountNumaGpu)->UseManualTime();

#endif // CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1