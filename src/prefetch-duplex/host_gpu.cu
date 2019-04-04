#if __CUDACC_VER_MAJOR__ >= 8

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

#define NAME "Comm_Prefetch_Duplex_HostGPU"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

auto Comm_Prefetch_Duplex_HostGPU = [] (benchmark::State &state,
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
  cudaStream_t stream0 = nullptr;
  cudaStream_t stream1 = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEvent_t other = nullptr;
  char *ptrs[2] = {nullptr};

#if USE_NUMA
  numa_bind_node(numa_id);
#endif

  OR_SKIP(utils::cuda_reset_device(cuda_id));
  OR_SKIP(cudaSetDevice(cuda_id));

  // one stream for h2d, one stream for d2h
  OR_SKIP(cudaStreamCreate(&stream0));
  OR_SKIP(cudaStreamCreate(&stream1));
  OR_SKIP(cudaEventCreate(&start));
  OR_SKIP(cudaEventCreate(&other));
  OR_SKIP(cudaEventCreate(&stop));

  OR_SKIP(cudaMallocManaged(&ptrs[0], bytes));
  OR_SKIP(cudaMallocManaged(&ptrs[1], bytes));
  OR_SKIP(cudaMemset(ptrs[0], 0, bytes));
  OR_SKIP(cudaMemset(ptrs[1], 0, bytes));
  OR_SKIP(cudaDeviceSynchronize());
  
  for (auto _ : state) {
    OR_SKIP(cudaMemPrefetchAsync(ptrs[0], bytes, cudaCpuDeviceId));
    flush_all(ptrs[0], bytes);
    OR_SKIP(cudaMemPrefetchAsync(ptrs[1], bytes, cuda_id));

    OR_SKIP(cudaEventRecord(start, stream0));
    OR_SKIP(cudaMemPrefetchAsync(ptrs[1], bytes, cudaCpuDeviceId, stream0));
    OR_SKIP(cudaEventRecord(other, stream1));
    OR_SKIP(cudaMemPrefetchAsync(ptrs[0], bytes, cuda_id, stream1));
    OR_SKIP(cudaStreamWaitEvent(stream0, other, 0 /*must be 0*/));
    OR_SKIP(cudaEventRecord(stop, stream0));

    OR_SKIP(cudaStreamSynchronize(stream0));
    float millis = 0;
    OR_SKIP(cudaEventElapsedTime(&millis, start, stop));
    state.SetIterationTime(millis / 1000);
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

  OR_SKIP(cudaEventDestroy(start));
  OR_SKIP(cudaEventDestroy(other));
  OR_SKIP(cudaEventDestroy(stop));
  OR_SKIP(cudaStreamDestroy(stream0));
  OR_SKIP(cudaStreamDestroy(stream1));

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
      benchmark::RegisterBenchmark(name.c_str(), Comm_Prefetch_Duplex_HostGPU,
#if USE_NUMA
        numa_id,
#endif // USE_NUMA
        cuda_id)->SMALL_ARGS()->UseManualTime();
#if USE_NUMA
    }
#endif // USE_NUMA
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
