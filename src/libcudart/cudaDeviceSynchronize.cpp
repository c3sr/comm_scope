/*! \file sync.cpp Measure the runtime cost of cudaDeviceSynchronize
 */

#include "scope/scope.hpp"

#define NAME "Comm_cudaDeviceSynchronize"

auto Comm_cudaDeviceSynchronize = [](benchmark::State &state, const int gpu,
                                     const int numaId) {
  numa::ScopedBind binder(numaId);

  // need each thread to set the device, but there's no way to reset
  // the device on a single thread first, because there's no thread barrier
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu), "");
  OR_SKIP_AND_RETURN(cudaFree(0), "failed to init");
  OR_SKIP_AND_RETURN(cudaDeviceSynchronize(), "failed to sync");

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
        benchmark::RegisterBenchmark(name.c_str(), Comm_cudaDeviceSynchronize,
                                     gpu, numaId)
            ->Threads(numThreads)
            ->UseRealTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
