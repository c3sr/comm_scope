#if CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1 && USE_OPENMP == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "utils/omp.hpp"

#include "args.hpp"

#define NAME "Comm/UM/Coherence/GPUToHostThreads"

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

static void Comm_UM_Coherence_GPUToHostThreads(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const size_t pageSize = page_size();

  const size_t threads = state.range(0);
  const auto bytes     = 1ULL << static_cast<size_t>(state.range(1));
  const int numa_id    = state.range(2);
  const int cuda_id    = state.range(3);

  omp_set_num_threads(threads);
  if (threads != omp_get_max_threads()) {
    state.SkipWithError(NAME " failed to set OMP threads");
    return;
  }

  omp_numa_bind_node(numa_id);

  if (PRINT_IF_ERROR(utils::cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA device");
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

  for (auto _ : state) {

    state.PauseTiming();

    cudaError_t err = cudaMemPrefetchAsync(ptr, bytes, cuda_id);
    if (cudaErrorInvalidDevice == err) {
      gpu_write<<<256, 256>>>(ptr, bytes, pageSize);
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }
    state.ResumeTiming();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bytes; i += pageSize) {
      benchmark::DoNotOptimize(ptr[i] = 0);
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  // reset to run on any node
  omp_numa_bind_node(-1);
}

// BENCHMARK(Comm_UM_Coherence_GPUToHostThreads)->Apply(ArgsThreadsCountNumaGpu)->UseRealTime();

#endif // CUDA_VERSION_MAJOR >= 8 && USE_NUMA == 1 && USE_OPENMP == 1