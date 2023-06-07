/* Measure the runtime cost of cudaMemcpy3DPeerAsync
 */

#include "scope/scope.hpp"

#define NAME "Comm_cudart_cudaMemcpyAsync_PinnedToGPU"

auto Comm_cudart_cudaMemcpyAsync_PinnedToGPU = [](benchmark::State &state,
                                                  const int gpu,
                                                  const int numa_id) {
  numa::ScopedBind binder(numa_id);

  OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu), "failed to reset CUDA device");

  // Create One stream per copy
  cudaStream_t stream = nullptr;
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), "failed to create stream");

  // fixed-size transfer
  const size_t bytes = 1024 * 10;

  void *src, *dst;

  // allocate on host
  src = aligned_alloc(page_size(), bytes);
  std::memset(src, 0, bytes);
  OR_SKIP_AND_RETURN(cudaHostRegister(src, bytes, cudaHostRegisterPortable),
                     "failed to register allocation");

  // allocate on gpu and enable peer access
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu), "failed to set device");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, bytes), "failed to perform cudaMalloc3D");
  OR_SKIP_AND_RETURN(cudaMemset(dst, 0, bytes),
                     "failed to perform src cudaMemset");

  for (auto _ : state) {
    // Start copy
    cudaError_t err =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);

    // measure one copy at a time
    state.PauseTiming();
    OR_SKIP_AND_BREAK(err, "failed to start cudaMemcpyAsync");

    OR_SKIP_AND_BREAK(cudaStreamSynchronize(stream), "failed to synchronize");
    state.ResumeTiming();
  }

  state.counters["gpu"] = gpu;
  state.counters["numa_id"] = numa_id;

  OR_SKIP_AND_RETURN(cudaStreamDestroy(stream), "cudaStreamDestroy");
  OR_SKIP_AND_RETURN(cudaHostUnregister(src), "");
  free(src);
  OR_SKIP_AND_RETURN(cudaFree(dst), "failed to cudaFree");
};

static void registerer() {
  std::string name;

  const std::set<int> numaNodes = numa::mems();
  const std::vector<MemorySpace> cudaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::cuda_device);

  for (int numa_id : numa::mems()) {
    for (size_t j = 0; j < cudaSpaces.size(); ++j) {
      auto gpu = cudaSpaces[j].device_id();
      name = std::string(NAME) + "/" + std::to_string(numa_id) + "/" +
             std::to_string(gpu);
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_cudart_cudaMemcpyAsync_PinnedToGPU, gpu, numa_id)
          ->UseRealTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
