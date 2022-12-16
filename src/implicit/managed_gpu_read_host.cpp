#include "args.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_managed_GPURdHost"

auto Comm_implicit_managed_GPURdHost =
    [](benchmark::State &state, const Device &dst_gpu,
       const MemorySpace &src_numa) {
       const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

      numa::ScopedBind sb(src_numa.numa_id());
      void *ptr;
      hipEvent_t start;
      hipEvent_t stop;

      if (PRINT_IF_ERROR(scope::hip_reset_device(dst_gpu.device_id()))) {
          state.SkipWithError(NAME " failed to reset hip src device");
          return;
      }

      if (PRINT_IF_ERROR(hipSetDevice(dst_gpu.device_id()))) {
          state.SkipWithError(NAME " failed to set hip write device");
          return;
      }

      if (PRINT_IF_ERROR(hipMallocManaged(&ptr, bytes))) {
          state.SkipWithError(NAME " failed to perform hipMallocManaged");
          return;
      }

      if (PRINT_IF_ERROR(hipMemset(ptr, 1, bytes))) {
          state.SkipWithError(NAME " failed to perform hipMemset");
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
        if (PRINT_IF_ERROR(hipMemPrefetchAsync(ptr, bytes, hipCpuDeviceId))) {
          state.SkipWithError(NAME "failed to prefetch to CPU");
          return; 
        }
        if (PRINT_IF_ERROR(hipDeviceSynchronize())) {
          state.SkipWithError(NAME "failed to sync");
          return; 
        }

        hipEventRecord(start);
        gpu_read<uint64_t><<<2048, 256>>>(ptr, nullptr, bytes);
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
      state.counters["src_numa"] = src_numa.numa_id();
      state.counters["dst_gpu"] = dst_gpu.device_id();
    };

static void registerer() {
  for (const Device &hip : scope::system::hip_devices()) {
    for (const MemorySpace &numa : scope::system::numa_memory_spaces()) {
      std::string name = std::string(NAME);
      name += "/" + std::to_string(numa.numa_id()) + "/" +
              std::to_string(hip.device_id());
      benchmark::RegisterBenchmark(
          name.c_str(), Comm_implicit_managed_GPURdHost, hip, numa)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
