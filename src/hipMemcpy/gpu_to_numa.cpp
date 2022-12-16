#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_hipMemcpy_GPUToNUMA"

auto Comm_hipMemcpy_GPUToNUMA = [](benchmark::State &state, const int numaId,
                                    const int hipId, const bool flush) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::bind_node(numaId);
  if (PRINT_IF_ERROR(scope::hip_reset_device(hipId))) {
    state.SkipWithError(NAME " failed to reset HIP device");
    return;
  }
  if (PRINT_IF_ERROR(hipSetDevice(hipId))) {
    state.SkipWithError(NAME " failed to set HIP device");
    return;
  }

  void *dst = aligned_alloc(page_size(), bytes);
  if (!dst) {
    state.SkipWithError(NAME " failed to perform aligned_alloc");
    return;
  }
  defer(free(dst));
  std::memset(dst, 0, bytes);
  void *src = nullptr;
  if (PRINT_IF_ERROR(hipMalloc(&src, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMalloc");
    return;
  }
  defer(hipFree(src));

  if (PRINT_IF_ERROR(hipMemset(src, 1, bytes))) {
    state.SkipWithError(NAME " failed to perform hipMemset");
    return;
  }

  for (auto _ : state) {
    std::memset(dst, 0, bytes);
    if (flush) {
      flush_all(dst, bytes);
    }

    auto start = scope::clock::now();
    hipError_t err =
        hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    double elapsed = scope::duration(scope::clock::now() - start).count();
    if (PRINT_IF_ERROR(err)) {
      state.SkipWithError(NAME " failed to perform memcpy");
      break;
    }
    state.SetIterationTime(elapsed);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["hip_id"] = hipId;
  state.counters["numa_id"] = numaId;

  // reset to run on any node
  numa::bind_node(-1);
};

static void registerer() {
  LOG(trace, NAME " registerer...");

  std::vector<MemorySpace> hipSpaces = scope::system::memory_spaces(MemorySpace::Kind::hip_device);
  std::vector<MemorySpace> numaSpaces = scope::system::memory_spaces(MemorySpace::Kind::numa);

  std::string name;
  for (const auto &hipSpace : hipSpaces) {
    for (const auto &numaSpace : numaSpaces) {

      const int hipId = hipSpace.device_id();
      const int numaId = numaSpace.numa_id();

      if (numa::can_execute_in_node(numaId)) {
        name = std::string(NAME) + "/" + std::to_string(numaId) + "/" +
              std::to_string(hipId);
        benchmark::RegisterBenchmark(name.c_str(), Comm_hipMemcpy_GPUToNUMA,
                                    numaId, hipId, false)
            ->SMALL_ARGS()
            ->UseManualTime();
        name = std::string(NAME) + "_flush/" + std::to_string(numaId) + "/" +
              std::to_string(hipId);
        benchmark::RegisterBenchmark(name.c_str(), Comm_hipMemcpy_GPUToNUMA,
                                    numaId, hipId, true)
            ->SMALL_ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
