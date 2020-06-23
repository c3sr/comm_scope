#if __CUDACC_VER_MAJOR__ >= 8

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_Prefetch_Duplex_HostGPU"

auto Comm_Prefetch_Duplex_HostGPU = [](benchmark::State &state,
                                       const int numa_id, const int cuda_id) {
  const size_t pageSize = page_size();
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  cudaStream_t stream0 = nullptr;
  cudaStream_t stream1 = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEvent_t other = nullptr;
  char *ptrs[2] = {nullptr};

  numa::ScopedBind binder(numa_id);

  OR_SKIP_AND_RETURN(cuda_reset_device(cuda_id), "");
  OR_SKIP_AND_RETURN(cudaSetDevice(cuda_id), "");

  // one stream for h2d, one stream for d2h
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream0), "");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream1), "");
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "");
  OR_SKIP_AND_RETURN(cudaEventCreate(&other), "");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "");

  OR_SKIP_AND_RETURN(cudaMallocManaged(&ptrs[0], bytes), "");
  OR_SKIP_AND_RETURN(cudaMallocManaged(&ptrs[1], bytes), "");
  OR_SKIP_AND_RETURN(cudaMemset(ptrs[0], 0, bytes), "");
  OR_SKIP_AND_RETURN(cudaMemset(ptrs[1], 0, bytes), "");
  OR_SKIP_AND_RETURN(cudaDeviceSynchronize(), "");

  for (auto _ : state) {
    OR_SKIP_AND_BREAK(cudaMemPrefetchAsync(ptrs[0], bytes, cudaCpuDeviceId),
                      "");
    flush_all(ptrs[0], bytes);
    OR_SKIP_AND_BREAK(cudaMemPrefetchAsync(ptrs[1], bytes, cuda_id), "");

    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream0), "");
    OR_SKIP_AND_BREAK(
        cudaMemPrefetchAsync(ptrs[1], bytes, cudaCpuDeviceId, stream0), "");
    OR_SKIP_AND_BREAK(cudaEventRecord(other, stream1), "");
    OR_SKIP_AND_BREAK(cudaMemPrefetchAsync(ptrs[0], bytes, cuda_id, stream1),
                      "");
    OR_SKIP_AND_BREAK(cudaStreamWaitEvent(stream0, other, 0 /*must be 0*/), "");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream0), "");

    OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream0), "");
    float millis = 0;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop), "");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;

  OR_SKIP_AND_RETURN(cudaEventDestroy(start), "");
  OR_SKIP_AND_RETURN(cudaEventDestroy(other), "");
  OR_SKIP_AND_RETURN(cudaEventDestroy(stop), "");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream0), "");
  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream1), "");

  for (auto p : ptrs) {
    OR_SKIP_AND_RETURN(cudaFree(p), "");
  }
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_id);
    if (!prop.concurrentManagedAccess) {
      LOG(debug,
          "{} can't run on device {}: requires concurrent managed access", NAME,
          cuda_id);
      continue;
    }

    for (auto numa_id : numa::ids()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numa_id) +
                         "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_Prefetch_Duplex_HostGPU,
                                   numa_id, cuda_id)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
