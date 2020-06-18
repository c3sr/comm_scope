#include "sysbench/sysbench.hpp"

#include "args.hpp"

#define NAME "Comm_UM_Malloc"

auto Comm_UM_Malloc = [](benchmark::State &state, const int numa_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  void *ptr = nullptr;

  for (auto _ : state) {
    auto start = std::chrono::system_clock::now();
    benchmark::DoNotOptimize(ptr = malloc(bytes));
    auto stop = std::chrono::system_clock::now();
    if (!ptr) {
      state.SkipWithError("failed to allocate");
    }
    free(ptr);
    double seconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
            .count();
    state.SetIterationTime(seconds / 1e9);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {
  for (auto numa_id : numa::ids()) {
    std::string name = std::string(NAME) + "/" + std::to_string(numa_id);
    benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Malloc, numa_id)
        ->BYTE_ARGS()
        ->UseManualTime();
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);
