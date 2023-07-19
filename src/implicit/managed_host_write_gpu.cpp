#include "args.hpp"

#include "scope/scope.hpp"

#include "../common/kernels.hpp"

#define NAME "Comm_implicit_managed_HostWrGPU"

auto Comm_implicit_managed_HostWrGPU =
    [](benchmark::State &state, const Device &src_gpu,
       const MemorySpace &dst_numa, const bool coarse,
       const bool flush) {
       const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind sb(dst_numa.numa_id());
  openmp::set_num_threads_to_numa_allowed_cpus();
  void *ptr;

  if (PRINT_IF_ERROR(scope::hip_reset_device(src_gpu.device_id()))) {
    state.SkipWithError(NAME " failed to reset hip src device");
    return;
  }

  if (PRINT_IF_ERROR(hipSetDevice(src_gpu.device_id()))) {
    state.SkipWithError(NAME " failed to set hip write device");
    return;
  }

  if (PRINT_IF_ERROR(hipMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMallocManaged");
    return;
  }

  const hipMemoryAdvise advice =
      coarse ? hipMemAdviseSetCoarseGrain : hipMemAdviseUnsetCoarseGrain;
  if (PRINT_IF_ERROR(hipMemAdvise(ptr, bytes, advice, src_gpu.device_id()))) {
    state.SkipWithError(NAME " failed to perform hipMemAdvise");
    return;
  }
  if (PRINT_IF_ERROR(hipMemAdvise(ptr, bytes, advice, hipCpuDeviceId))) {
    state.SkipWithError(NAME " failed to perform hipMemAdvise");
    return;
  }

      uint64_t iter = 1;
      for (auto _ : state) {
        if (flush) flush_all(ptr, bytes);
        if (PRINT_IF_ERROR(hipMemPrefetchAsync(ptr, bytes, src_gpu.device_id()))) {
          state.SkipWithError(NAME "failed to prefetch");
          return; 
        }
        if (PRINT_IF_ERROR(hipMemset(ptr, iter, bytes))) {
            state.SkipWithError(NAME " failed to perform hipMemset");
            return;
        }  
        if (PRINT_IF_ERROR(hipDeviceSynchronize())) {
          state.SkipWithError(NAME "failed to sync");
          return; 
        }

        auto start = scope::clock::now();
        cpu_write<uint64_t>(ptr, bytes, iter+1);
        auto stop = scope::clock::now();
        scope::duration elapsed = stop - start;
        state.SetIterationTime(elapsed.count());
      }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["dst_numa"] = dst_numa.numa_id();
  state.counters["src_gpu"] = src_gpu.device_id();
};

static void registerer() {

#if SCOPE_USE_OPENMP == 1

  for (const Device &hip : scope::system::hip_devices()) {
    for (const MemorySpace &numa : scope::system::numa_memory_spaces()) {
      for (const bool coarse : {true, false}) {
        for (const bool flush : {true, false}) {
          std::string name = std::string(NAME);
          name += std::string("_") + (coarse ? "coarse" : "fine");
          if (flush) name += "_flush";
          name += "/" + std::to_string(numa.numa_id()) + "/" +
                  std::to_string(hip.device_id());
          benchmark::RegisterBenchmark(
              name.c_str(), Comm_implicit_managed_HostWrGPU, hip, numa, coarse, flush)
              ->SMALL_ARGS()
              ->UseManualTime();
        }
      }
    }
  }

#else
#warning "benchmark disabled: SCOPE_USE_OPENMP != 1
#endif // SCOPE_USE_OPENMP==1
}

SCOPE_AFTER_INIT(registerer, NAME);
