#include "args.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_mapped_HostWrGPU"

auto Comm_implicit_mapped_HostWrGPU =
    [](benchmark::State &state, const Device &own_gpu,
       const MemorySpace &wr_numa, bool flush) {
      const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

      void *ptr = nullptr;
      void *dptr = nullptr;

      numa::ScopedBind sb(wr_numa.numa_id());
      if (PRINT_IF_ERROR(scope::hip_reset_device(own_gpu.device_id()))) {
        state.SkipWithError(NAME " failed to reset hip src device");
        return;
      }

      if (PRINT_IF_ERROR(hipSetDevice(own_gpu.device_id()))) {
        state.SkipWithError(NAME " failed to set hip write device");
        return;
      }
      
      if (PRINT_IF_ERROR(hipMalloc(&ptr, bytes))) {
        state.SkipWithError(NAME " failed to perform hipMalloc");
        return;
      }
      defer(hipFree(ptr));
      
      // if (PRINT_IF_ERROR(hipHostGetDevicePointer(&dptr, ptr, 0))) {
      //   state.SkipWithError(NAME " failed to perform hipHostGetDevicePointer");
      //   return;
      // }

      for (auto _ : state) {

        if (PRINT_IF_ERROR(hipMemset(ptr, 1, bytes))) {
          state.SkipWithError(NAME " failed to perform hipMemset");
          return;
        }
        if (flush)
          flush_all(ptr, bytes);

        auto start = scope::clock::now();
        cpu_write<uint64_t>(dptr, bytes);
        auto stop = scope::clock::now();
        state.SetIterationTime(scope::duration(stop-start).count());
      }

      state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
      state.counters["bytes"] = bytes;
      state.counters["wr_numa"] = wr_numa.numa_id();
      state.counters["own_gpu"] = own_gpu.device_id();
    };

static void registerer() {
  for (const Device &hip : scope::system::hip_devices()) {
    if (hip.can_map_host_memory()) {
      for (const MemorySpace &numa : scope::system::numa_memory_spaces()) {
        for (const bool flush : {false, true}) {
          std::string name = std::string(NAME);
          if (flush) {
            name += "_flush";
          }
          name += "/" + std::to_string(numa.numa_id()) + "/" +
                  std::to_string(hip.device_id());
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_implicit_mapped_HostWrGPU, hip, numa, flush)
              ->SMALL_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
