#if __CUDACC_VER_MAJOR__ >= 8

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_UM_Demand_GPUToHost"

static void cpu_write(char *ptr, const size_t n, const size_t stride) {
  for (size_t i = 0; i < n; i += stride) {
    benchmark::DoNotOptimize(ptr[i] = 0);
  }
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
  size_t wx = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

auto Comm_UM_Demand_GPUToHost = [](benchmark::State &state, const int numa_id,
                                   const int cuda_id) {
  const size_t pageSize = page_size();

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  if (PRINT_IF_ERROR(cuda_reset_device(cuda_id))) {
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

    cpu_write(ptr, bytes, pageSize);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : numa::mems()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numa_id) +
                         "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Demand_GPUToHost,
                                   numa_id, cuda_id)
          ->SMALL_ARGS();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);

#endif // __CUDACC_VER_MAJOR__ >= 8
