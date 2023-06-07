#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_libc_malloc_memset"

auto Comm_libc_malloc_memset = [](benchmark::State &state, const int numa_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  void *ptr = nullptr;

  for (auto _ : state) {
    auto start = std::chrono::system_clock::now();
    benchmark::DoNotOptimize(ptr = malloc(bytes));
    std::memset(ptr, 0, bytes);
    benchmark::ClobberMemory();
    auto stop = std::chrono::system_clock::now();
    if (!ptr) {
      state.SkipWithError(NAME " failed to allocate");
      return;
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

  const std::vector<MemorySpace> numaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (const auto &numa : numaSpaces) {

    const auto numa_id = numa.numa_id();
    std::string name = std::string(NAME) + "/" + std::to_string(numa_id);
    benchmark::RegisterBenchmark(name.c_str(), Comm_libc_malloc_memset, numa_id)
        ->BYTE_ARGS()
        ->UseManualTime();
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
