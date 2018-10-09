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
#include "utils/cache_control.hpp"

#define NAME "Comm_Coherence_Duplex_HostGPU"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

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

auto Comm_Coherence_Duplex_HostGPU = [] (benchmark::State &state,
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

  OR_SKIP(utils::cuda_reset_device(cuda_id));
  OR_SKIP(cudaSetDevice(cuda_id));

  char *ptrs[2] = {nullptr};
  OR_SKIP(cudaMallocManaged(&ptrs[0], bytes));
  OR_SKIP(cudaMallocManaged(&ptrs[1], bytes));
  OR_SKIP(cudaMemset(ptrs[0], 0, bytes));
  OR_SKIP(cudaMemset(ptrs[1], 0, bytes));
  OR_SKIP(cudaDeviceSynchronize());
  

  for (auto _ : state) {
    state.PauseTiming();

    cudaError_t err;
    // move ptrs[0] to cpu
    err = cudaMemPrefetchAsync(ptrs[0], bytes, cudaCpuDeviceId);
    if (err == cudaErrorInvalidDevice) {
      for (size_t i = 0; i < bytes; i += pageSize) {
        ptrs[0][i] = 0;
      }
    }
    flush_all(ptrs[0], bytes);


    // move ptrs[1] to gpu
    err = cudaMemPrefetchAsync(ptrs[1], bytes, cuda_id);
    if (err == cudaErrorInvalidDevice) {
      gpu_write<<<256, 256>>>(ptrs[1], bytes, pageSize);
    }
    OR_SKIP(cudaDeviceSynchronize());

    state.ResumeTiming();
    // access ptrs[0] on gpu
    gpu_write<<<256, 256>>>(ptrs[0], bytes, pageSize);
    // access prts[1] on cpu
    for (size_t i = 0; i < bytes; i += pageSize) {
      ptrs[1][i] = 0;
    }
    OR_SKIP(cudaDeviceSynchronize());
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
#if USE_NUMA
  state.counters["numa_id"] = numa_id;
#endif // USE_NUMA

#if USE_NUMA
  numa_bind_node(-1);
#endif

  for (auto p : ptrs) {
    OR_SKIP(cudaFree(p));
  }

};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_id);
    if (!prop.concurrentManagedAccess) {
      LOG(debug, "{} can't run on device {}: requires concurrent managed access", NAME, cuda_id);
      continue;
    }

#if USE_NUMA
    for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_Coherence_Duplex_HostGPU,
#if USE_NUMA
        numa_id,
#endif // USE_NUMA
        cuda_id)->SMALL_ARGS();
#if USE_NUMA
    }
#endif // USE_NUMA
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // CUDA_VERSION_MAJOR >= 8
