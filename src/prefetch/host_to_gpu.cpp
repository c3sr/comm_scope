#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_hipManaged_Prefetch_HostToGPU"

auto Comm_hipManaged_Prefetch_HostToGPU = [](benchmark::State &state,
                                     const int numa_id,
                                     const int hip_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  if (PRINT_IF_ERROR(scope::hip_reset_device(hip_id))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  if (PRINT_IF_ERROR(hipSetDevice(hip_id))) {
    state.SkipWithError(NAME " failed to set hip dst device");
    return;
  }

  char *ptr = nullptr;
  if (PRINT_IF_ERROR(hipMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMallocManaged");
    return;
  }
  defer(hipFree(ptr));

  if (PRINT_IF_ERROR(hipMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  hipEvent_t start, stop;
  if (PRINT_IF_ERROR(hipEventCreate(&start))) {
    state.SkipWithError(NAME " failed to create start event");
    return;
  }
  defer(hipEventDestroy(start));

  if (PRINT_IF_ERROR(hipEventCreate(&stop))) {
    state.SkipWithError(NAME " failed to create end event");
    return;
  }
  defer(hipEventDestroy(stop));

  for (auto _ : state) {
    if (PRINT_IF_ERROR(hipMemPrefetchAsync(ptr, bytes, hipCpuDeviceId))) {
      state.SkipWithError(NAME " failed to move data to src");
      return;
    }
    if (PRINT_IF_ERROR(hipDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }

    hipEventRecord(start);
    if (PRINT_IF_ERROR(hipMemPrefetchAsync(ptr, bytes, hip_id))) {
      state.SkipWithError(NAME " failed to move data to src");
      return;
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float millis = 0;
    if (PRINT_IF_ERROR(hipEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["hip_id"] = hip_id;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {

  std::vector<MemorySpace> numaSpaces = scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (auto hip : scope::system::hip_devices()) {
    for (auto ns : numaSpaces) {
      const int numa_id = ns.numa_id();
      if (numa::can_execute_in_node(numa_id)) {
        std::string name = std::string(NAME) + "/" + std::to_string(numa_id) +
                          "/" + std::to_string(hip.device_id());
        benchmark::RegisterBenchmark(name.c_str(), Comm_hipManaged_Prefetch_HostToGPU,
                                    numa_id, hip.device_id())
            ->SMALL_ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
