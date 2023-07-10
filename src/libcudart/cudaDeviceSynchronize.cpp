/*! \file sync.cpp Measure the runtime cost of cudaDeviceSynchronize
 */

#include "scope/scope.hpp"

#define NAME "Comm_cudaDeviceSynchronize"

auto Comm_cudart_kernel = [](benchmark::State &state, const int gpu,
                             const int numaId) {
  numa::ScopedBind binder(numaId);

  if (0 == state.thread_index()) {
    OR_SKIP_AND_RETURN(scope::cuda_reset_device(gpu), "failed to reset CUDA device");
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu), "");
  OR_SKIP_AND_RETURN(cudaFree(0), "failed to init");

  cudaError_t err = cudaSuccess;
  for (auto _ : state) {
    err = cudaDeviceSynchronize();
  }

  OR_SKIP_AND_RETURN(err, "failed to lsync");

  state.SetItemsProcessed(state.iterations());
  state.counters["gpu"] = gpu;
};

static void registerer() {
  std::string name;
  const std::vector<Device> cudas = scope::system::cuda_devices();
  for (size_t i = 0; i < cudas.size(); ++i) {
    for (int numaId : numa::mems()) {
      for (size_t numThreads = 1;
           numThreads <= numa::cpus_in_node(numaId).size(); numThreads *= 2) {
        int gpu = cudas[i];
        name = std::string(NAME) + "/" + std::to_string(numaId) + "/" +
               std::to_string(gpu);
        benchmark::RegisterBenchmark(name.c_str(), Comm_cudart_kernel, gpu,
                                     numaId)
            ->Threads(numThreads)
            ->UseRealTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
