/*! \file sync.cpp Measure the runtime cost of hipDeviceSynchronize
 */

#include "scope/scope.hpp"

#define NAME "Comm_hipDeviceSynchronize"

auto Comm_hipDeviceSynchronize = [](benchmark::State &state, const int gpu,
                                    const int numaId) {
  numa::ScopedBind binder(numaId);

  if (0 == state.thread_index()) {
    OR_SKIP_AND_RETURN(scope::hip_reset_device(gpu),
                       "failed to reset HIP device");
  }

  OR_SKIP_AND_RETURN(hipSetDevice(gpu), "");
  OR_SKIP_AND_RETURN(hipFree(0), "failed to init");

  hipError_t err = hipSuccess;
  for (auto _ : state) {
    err = hipDeviceSynchronize();
  }

  OR_SKIP_AND_RETURN(err, "failed to lsync");

  state.SetItemsProcessed(state.iterations());
  state.counters["gpu"] = gpu;
};

static void registerer() {
  std::string name;
  const std::vector<Device> hips = scope::system::hip_devices();
  for (size_t i = 0; i < hips.size(); ++i) {
    for (int numaId : numa::mems()) {
      for (size_t numThreads = 1;
           numThreads <= numa::cpus_in_node(numaId).size(); numThreads *= 2) {
        int gpu = hips[i];
        name = std::string(NAME) + "/" + std::to_string(numaId) + "/" +
               std::to_string(gpu);
        benchmark::RegisterBenchmark(name.c_str(), Comm_hipDeviceSynchronize,
                                     gpu, numaId)
            ->Threads(numThreads)
            ->UseRealTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
