#if __CUDACC_VER_MAJOR__ >= 8

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_Demand_Duplex_HostGPU"

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
  size_t wx = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

auto Comm_Demand_Duplex_HostGPU = [](benchmark::State &state, const int numa_id,
                                     const int cuda_id) {
  const size_t pageSize = page_size();
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  OR_SKIP_AND_RETURN(cuda_reset_device(cuda_id), "");
  OR_SKIP_AND_RETURN(cudaSetDevice(cuda_id), "");

  char *ptrs[2] = {nullptr};
  OR_SKIP_AND_RETURN(cudaMallocManaged(&ptrs[0], bytes), "");
  OR_SKIP_AND_RETURN(cudaMallocManaged(&ptrs[1], bytes), "");
  OR_SKIP_AND_RETURN(cudaMemset(ptrs[0], 0, bytes), "");
  OR_SKIP_AND_RETURN(cudaMemset(ptrs[1], 0, bytes), "");
  OR_SKIP_AND_RETURN(cudaDeviceSynchronize(), "");

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
    OR_SKIP_AND_BREAK(cudaDeviceSynchronize(), "");

    state.ResumeTiming();
    // access ptrs[0] on gpu
    gpu_write<<<256, 256>>>(ptrs[0], bytes, pageSize);
    // access prts[1] on cpu
    for (size_t i = 0; i < bytes; i += pageSize) {
      ptrs[1][i] = 0;
    }
    OR_SKIP_AND_BREAK(cudaDeviceSynchronize(), "");
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;

  for (auto p : ptrs) {
    OR_SKIP_AND_RETURN(cudaFree(p), "");
  }
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, cuda_id);
    if (err != cudaSuccess) {
      LOG(error, "error getting device props in {}", NAME);
      continue;
    }
    if (!prop.concurrentManagedAccess) {
      LOG(debug,
          "device {} doesn't support {}: requires concurrent managed access",
          cuda_id, NAME);
      continue;
    }

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, cuda_id);
    // if (true) {
    //   LOG(debug, "device {} doesn't support {}: requires concurrent managed
    //   access", cuda_id, NAME); continue;
    // }

    for (auto numa_id : numa::ids()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numa_id) +
                         "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_Demand_Duplex_HostGPU,
                                   numa_id, cuda_id)
          ->SMALL_ARGS();
    }
  }
  LOG(debug, "Done after_init for {}", NAME);
}

SCOPE_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
