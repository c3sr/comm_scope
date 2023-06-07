#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyAsync_HostToGPU"

auto Comm_cudaMemcpyAsync_HostToGPU = [](benchmark::State &state,
                                         const int numa_id, const int cuda_id,
                                         const bool flush) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::bind_node(numa_id);
  if (PRINT_IF_ERROR(scope::cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }

  void *src = aligned_alloc(page_size(), bytes);
  if (!src) {
    state.SkipWithError(NAME " failed to performance aligned_alloc");
    return;
  }
  defer(free(src));
  std::memset(src, 0, bytes);
  void *dst = nullptr;
  if (PRINT_IF_ERROR(cudaMalloc(&dst, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(dst));

  if (PRINT_IF_ERROR(cudaMemset(dst, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    std::memset(src, 0, bytes);
    if (flush) {
      flush_all(src, bytes);
    }
    cudaEventRecord(start, NULL);
    const auto cuda_err =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError(NAME " failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;

  // reset to run on any node
  numa::bind_node(-1);
};

static void registerer() {
  std::string name;
  const std::vector<MemorySpace> cudaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::cuda_device);
  const std::vector<MemorySpace> numaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (const auto &cudaSpace : cudaSpaces) {
    for (const auto &numaSpace : numaSpaces) {

      const int cudaId = cudaSpace.device_id();
      const int numaId = numaSpace.numa_id();
      name = std::string(NAME) + "/" + std::to_string(numaId) + "/" +
             std::to_string(cudaId);
      benchmark::RegisterBenchmark(name.c_str(), Comm_cudaMemcpyAsync_HostToGPU,
                                   numaId, cudaId, false)
          ->SMALL_ARGS()
          ->UseManualTime();
      name = std::string(NAME) + "_flush/" + std::to_string(numaId) + "/" +
             std::to_string(cudaId);
      benchmark::RegisterBenchmark(name.c_str(), Comm_cudaMemcpyAsync_HostToGPU,
                                   numaId, cudaId, true)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
