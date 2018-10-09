#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#endif // USE_NUMA

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "zero-copy/args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"
#include "init/numa.hpp"
#include "utils/cache_control.hpp"


#define NAME "Comm_ZeroCopy_HostGPU"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

typedef enum {
  READ,
  WRITE,
} AccessType;

std::string to_string(const AccessType &a) {
  if (a == READ) {
    return "_Read";
  } else {
    return "_Write";
  }
}

// typedef enum {
//   FLUSH,
//   NO_FLUSH,
// } FlushType;

// static std::string to_string(const FlushType &a) {
//   if (a == FLUSH) {
//     return "_Flush";
//   } else {
//     return "";
//   }
// }

template <typename write_t>
__global__ void gpu_write(write_t *ptr, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(write_t);

  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    ptr[i] = 0;
  }
}


template <typename read_t>
__global__ void gpu_read(const read_t *ptr, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(read_t);

  __shared__ int32_t s[256];
  int32_t t;

  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    t += ptr[i];
  }
  s[threadIdx.x] = t;
  (void) s[threadIdx.x];
}


auto Comm_ZeroCopy_HostGPU = [](benchmark::State &state, const int src_numa, const int dst_cuda, const AccessType access_type) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));

#if USE_NUMA
  numa_bind_node(src_numa);
#endif

  OR_SKIP(utils::cuda_reset_device(dst_cuda));
  OR_SKIP(cudaSetDevice(dst_cuda));

  void *ptr = aligned_alloc(pageSize, bytes);
  defer(free(ptr));
  if (!ptr && bytes) {
    state.SkipWithError(NAME " failed to allocate host memory");
    return;
  }
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

    OR_SKIP(cudaEventRecord(start));
    if (READ == access_type) {
      gpu_read<int32_t><<<256, 256>>>((int32_t*) dptr, bytes);
    } else {
      gpu_write<int32_t><<<256, 256>>>((int32_t *)dptr, bytes);
    }

    OR_SKIP(cudaEventRecord(stop));
    OR_SKIP(cudaEventSynchronize(stop));

    float millis = 0;
    OR_SKIP(cudaEventElapsedTime(&millis, start, stop));
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
#if USE_NUMA
  state.counters["src_numa"] = src_numa;
#endif // USE_NUMA
  state.counters["dst_cuda"] = dst_cuda;

#if USE_NUMA
  numa_bind_node(-1);
#endif
};

static void registerer() {

  for (auto workload : {READ, WRITE}) {



  for (auto cuda_id : unique_cuda_device_ids()) {
#if USE_NUMA
    for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME) + to_string(workload)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_HostGPU,
#if USE_NUMA
        numa_id,
#endif // USE_NUMA
        cuda_id, workload)->ARGS()->UseManualTime();
#if USE_NUMA
    }
#endif // USE_NUMA
  }
}
}

SCOPE_REGISTER_AFTER_INIT(registerer);
