#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudart_cudaMallocManaged"

auto Comm_cudart_cudaMallocManaged = [](benchmark::State &state,
                                        const int numa_id, const int cuda_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  OR_SKIP_AND_RETURN(cuda_reset_device(cuda_id), "failed to reset device");
  OR_SKIP_AND_RETURN(cudaSetDevice(cuda_id), "failed to set CUDA dst device");

  char *ptr = nullptr;

  cudaError_t err;
  for (auto _ : state) {
    err = cudaMallocManaged(&ptr, bytes);
    state.PauseTiming();
    OR_SKIP_AND_BREAK(err, "");
    OR_SKIP_AND_BREAK(cudaFree(ptr), "");
    state.ResumeTiming();
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
      benchmark::RegisterBenchmark(name.c_str(), Comm_cudart_cudaMallocManaged,
                                   numa_id, cuda_id)
          ->ALLOC_ARGS();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
