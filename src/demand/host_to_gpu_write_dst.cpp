#include "args.hpp"
#include "demand.hpp"

#define NAME "Comm_hipManaged_HostToGPUWriteDst"

constexpr int CACHE_LINE_SIZE = 64;

auto Comm_hipManaged_HostToGPUWriteDst = [](benchmark::State &state, const int src_numa,
                                  const int dst_gpu) {


  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  UnaryData data = setup<Kind::HostToGPU>(state, NAME, bytes, src_numa, dst_gpu);
  defer(hipFree(data.ptr));
  defer(hipEventDestroy(data.start));
  defer(hipEventDestroy(data.stop));
  if (data.error) {
    return;
  }


  for (auto _ : state) {
    prep_iteration<Kind::HostToGPU>(data.ptr, bytes, src_numa, dst_gpu);
    if (PRINT_IF_ERROR(hipGetLastError())) {
      state.SkipWithError(NAME " failed to prep iteration");
      return;
    }

    hipEventRecord(data.start);
    gpu_write<<<2048, 256>>>(data.ptr, bytes, CACHE_LINE_SIZE);
    hipEventRecord(data.stop);
    if (PRINT_IF_ERROR(hipEventSynchronize(data.stop))) {
      state.SkipWithError(NAME " failed to do kernels");
      return;
    }

    float millis = 0;
    if (PRINT_IF_ERROR(hipEventElapsedTime(&millis, data.start, data.stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_numa"] = src_numa;
  state.counters["dst_gpu"] = dst_gpu;
};

static void registerer() {

  std::vector<MemorySpace> hipSpaces = scope::system::memory_spaces(MemorySpace::Kind::hip_device);
  std::vector<MemorySpace> numaSpaces = scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (const MemorySpace &ns : numaSpaces) {
    for (const MemorySpace &hs : hipSpaces) {
      auto src_numa = ns.numa_id();
      auto dst_gpu = hs.device_id();
      if (numa::can_execute_in_node(src_numa)) {
        std::string name = std::string(NAME) + "/" + std::to_string(src_numa) +
                          "/" + std::to_string(dst_gpu);
        benchmark::RegisterBenchmark(name.c_str(), Comm_hipManaged_HostToGPUWriteDst,
                                    src_numa, dst_gpu)
            ->SMALL_ARGS()
            ->UseManualTime();
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);


