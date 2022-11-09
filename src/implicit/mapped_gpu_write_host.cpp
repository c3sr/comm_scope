#include "args.hpp"
#include "implicit.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_mapped_GPUWrHost"

constexpr int CACHE_LINE_SIZE = 64;

auto Comm_implicit_mapped_GPUWrHost =
    [](benchmark::State &state, const Device &wr_gpu,
       const MemorySpace &own_numa, bool flush) {
      const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

      void *ptr = nullptr;
      void *dptr = nullptr;
      hipEvent_t start;
      hipEvent_t stop;

      numa::ScopedBind sb(own_numa.numa_id());
      if (PRINT_IF_ERROR(scope::hip_reset_device(wr_gpu.device_id()))) {
        state.SkipWithError(NAME " failed to reset hip src device");
        return;
      }

      if (PRINT_IF_ERROR(hipSetDevice(wr_gpu.device_id()))) {
        state.SkipWithError(NAME " failed to set hip write device");
        return;
      }

#if 0
  // this didn't work on MI25
  ptr = (char*)aligned_alloc(page_size(), bytes);  
  if (PRINT_IF_ERROR(hipHostRegister(&ptr, bytes, hipHostRegisterMapped))) {
    state.SkipWithError(NAME " failed to perform hipHostRegister");
    return;
  }
  defer(hipHostUnregister(ptr));
  defer(free(ptr));
#else
      if (PRINT_IF_ERROR(hipHostMalloc(&ptr, bytes, 0))) {
        state.SkipWithError(NAME " failed to perform hipHostMalloc");
        return;
      }
      defer(hipHostFree(ptr));
#endif
      std::memset(ptr, 1, bytes);

      if (PRINT_IF_ERROR(hipHostGetDevicePointer(&dptr, ptr, 0))) {
        state.SkipWithError(NAME " failed to perform hipHostGetDevicePointer");
        return;
      }

      if (PRINT_IF_ERROR(hipEventCreate(&start))) {
        state.SkipWithError(NAME " failed to create start event");
        return;
      }
      defer(hipEventDestroy(start));

      if (PRINT_IF_ERROR(hipEventCreate(&stop))) {
        state.SkipWithError(NAME " failed to create stop event");
        return;
      }
      defer(hipEventDestroy(stop));

      for (auto _ : state) {

        std::memset(ptr, 1, bytes);
        if (flush)
          flush_all(ptr, bytes);

        hipEventRecord(start);
        gpu_write<<<2048, 256>>>((char *)dptr, bytes, CACHE_LINE_SIZE);
        hipEventRecord(stop);
        if (PRINT_IF_ERROR(hipEventSynchronize(stop))) {
          state.SkipWithError(NAME " failed to do kernels");
          return;
        }

        float millis = 0;
        if (PRINT_IF_ERROR(hipEventElapsedTime(&millis, start, stop))) {
          state.SkipWithError(NAME " failed to get elapsed time");
          break;
        }
        state.SetIterationTime(millis / 1000);
      }

      state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
      state.counters["bytes"] = bytes;
      state.counters["own_numa"] = own_numa.numa_id();
      state.counters["wr_gpu"] = wr_gpu.device_id();
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
          name += "/" + std::to_string(hip.device_id()) + "/" +
                  std::to_string(numa.numa_id());
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_implicit_mapped_GPUWrHost, hip, numa, flush)
              ->SMALL_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
