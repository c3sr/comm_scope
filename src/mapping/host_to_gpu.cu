#include <cassert>

#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#endif // USE_NUMA

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "mapping/args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"

#define NAME "Comm/Mapping/HostToGPU"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

template <bool NOOP = false>
__global__ void gpu_write(void *ptr, const size_t count, const size_t stride) {
  if (NOOP) {
    return;
  }

  char *p = (char *) ptr;

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx             = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      p[i] = 0;
    }
  }
}

static void Comm_Mapping_HostToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));
  const int dst_cuda  = FLAG(cuda_device_ids)[0];
#if USE_NUMA
  const int src_numa = FLAG(numa_ids)[0];
#endif

#if USE_NUMA
  numa_bind_node(src_numa);
#endif

  OR_SKIP(utils::cuda_reset_device(dst_cuda));
  OR_SKIP(cudaSetDevice(dst_cuda));

  void *ptr = aligned_alloc(pageSize, bytes);
  defer(free(ptr));
  std::memset(ptr, 0, bytes);

  OR_SKIP(cudaHostRegister(ptr, bytes, cudaHostRegisterMapped));
  defer(cudaHostUnregister(ptr));

  // get a valid device pointer
  void *dptr;
  cudaDeviceProp prop;
  OR_SKIP(cudaGetDeviceProperties(&prop, dst_cuda));
  if (prop.canUseHostPointerForRegisteredMem) {
    dptr = ptr;
  } else {
    OR_SKIP(cudaHostGetDevicePointer(&dptr, ptr, 0));
  }


  cudaEvent_t start, stop;
  OR_SKIP(cudaEventCreate(&start));
  defer(cudaEventDestroy(start));
  OR_SKIP(cudaEventCreate(&stop));
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {
    cudaEventRecord(start);
    gpu_write<<<256, 256>>>(dptr, bytes, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millis = 0;
    OR_SKIP(cudaEventElapsedTime(&millis, start, stop));
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_numa"] = src_numa;
  state.counters["dst_cuda"] = dst_cuda;

#if USE_NUMA
  numa_bind_node(-1);
#endif
}

BENCHMARK(Comm_Mapping_HostToGPU)->SMALL_ARGS()->UseManualTime();
