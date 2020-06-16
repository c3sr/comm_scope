#include <cassert>

 #include "sysbench/sysbench.hpp"
 

#include "args.hpp"

#define NAME "Comm_UM_Malloc_Memset"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

auto Comm_UM_Malloc_Memset = [] (benchmark::State &state
  #if USE_NUMA
  , const int numa_id
  #endif // USE_NUMA
  ) {

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

#if USE_NUMA
  numa::bind_node(numa_id);
#endif

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
    double seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
    state.SetIterationTime(seconds / 1e9);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
#if USE_NUMA
  state.counters["numa_id"] = numa_id;
#endif // USE_NUMA

#if USE_NUMA
  numa::bind_node(-1);
#endif
};

static void registerer() {
#if USE_NUMA
    for (auto numa_id : numa::ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
      ;
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Malloc_Memset
#if USE_NUMA
        ,numa_id
#endif // USE_NUMA
        )->BYTE_ARGS()->UseManualTime();
#if USE_NUMA
    }
#endif // USE_NUMA
}

SYSBENCH_AFTER_INIT(registerer, NAME);


