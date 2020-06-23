#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_libc_malloc"

auto Comm_libc_malloc = [](benchmark::State &state, const int numa_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::ScopedBind binder(numa_id);

  void *ptr = nullptr;

  for (auto _ : state) {
    benchmark::DoNotOptimize(ptr = malloc(bytes));
    state.PauseTiming();
    if (!ptr) {
      state.SkipWithError("failed to allocate");
    }
    free(ptr);
    state.ResumeTiming();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["numa_id"] = numa_id;
};

static void registerer() {
  for (auto numa_id : numa::ids()) {
    std::string name = std::string(NAME) + "/" + std::to_string(numa_id);
    benchmark::RegisterBenchmark(name.c_str(), Comm_libc_malloc, numa_id)
        ->BYTE_ARGS();
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
