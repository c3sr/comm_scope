#if CUDA_VERSION_MAJOR >= 8

#include <cassert>

#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#endif // USE_NUMA

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"
#include "init/numa.hpp"

#define NAME "Comm_UM_Coherence_HostToGPU"

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

auto Comm_UM_Coherence_HostToGPU = [] (benchmark::State &state,
  #if USE_NUMA
  const int numa_id,
  #endif // USE_NUMA
  const int cuda_id) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));

#if USE_NUMA
  numa_bind_node(numa_id);
#endif

  if (PRINT_IF_ERROR(utils::cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
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
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
#if USE_NUMA
  state.counters["numa_id"] = numa_id;
#endif // USE_NUMA

#if USE_NUMA
  numa_bind_node(-1);
#endif
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
#if USE_NUMA
    for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Coherence_HostToGPU,
#if USE_NUMA
        numa_id,
#endif // USE_NUMA
        cuda_id)->SMALL_ARGS()->UseManualTime();
#if USE_NUMA
    }
#endif // USE_NUMA
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // CUDA_VERSION_MAJOR >= 8
