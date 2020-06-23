#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_ZeroCopy_HostToGPU"

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
  (void)s[threadIdx.x];
}

auto Comm_ZeroCopy_HostToGPU = [](benchmark::State &state, const int src_numa,
                                  const int dst_cuda,
                                  const AccessType access_type) {
  const size_t pageSize = page_size();

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::bind_node(src_numa);

  OR_SKIP_AND_RETURN(cuda_reset_device(dst_cuda), "");
  OR_SKIP_AND_RETURN(cudaSetDevice(dst_cuda), "");

  void *ptr = aligned_alloc(pageSize, bytes);
  defer(free(ptr));
  if (!ptr && bytes) {
    state.SkipWithError(NAME " failed to allocate host memory");
    return;
  }
  std::memset(ptr, 0, bytes);

  OR_SKIP_AND_RETURN(cudaHostRegister(ptr, bytes, cudaHostRegisterMapped), "");
  defer(cudaHostUnregister(ptr));

  // get a valid device pointer
  void *dptr;
  cudaDeviceProp prop;
  OR_SKIP_AND_RETURN(cudaGetDeviceProperties(&prop, dst_cuda), "");

#if __CUDACC_VER_MAJOR__ >= 9
  if (prop.canUseHostPointerForRegisteredMem) {
#else
  if (false) {
#endif
    dptr = ptr;
  } else {
    OR_SKIP_AND_RETURN(cudaHostGetDevicePointer(&dptr, ptr, 0), "");
  }

  cudaEvent_t start, stop;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "");
  defer(cudaEventDestroy(start));
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "");
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {

    OR_SKIP_AND_BREAK(cudaEventRecord(start), "");
    if (READ == access_type) {
      gpu_read<int32_t><<<256, 256>>>((int32_t *)dptr, bytes);
    } else {
      gpu_write<int32_t><<<256, 256>>>((int32_t *)dptr, bytes);
    }

    OR_SKIP_AND_BREAK(cudaEventRecord(stop), "");
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "");

    float millis = 0;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop), "");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_numa"] = src_numa;
  state.counters["dst_cuda"] = dst_cuda;

  numa::bind_node(-1);
};

static void registerer() {

  for (auto workload : {READ, WRITE}) {

    for (auto cuda_id : unique_cuda_device_ids()) {
      for (auto numa_id : numa::ids()) {
        std::string name = std::string(NAME) + to_string(workload) + "/" +
                           std::to_string(numa_id) + "/" +
                           std::to_string(cuda_id);
        benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_HostToGPU,
                                     numa_id, cuda_id, workload)
            ->ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
