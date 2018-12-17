#include <cassert>

#include <cuda_runtime.h>
#if USE_NUMA
#include <numa.h>
#endif // USE_NUMA

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "utils/numa.hpp"
#include "init/numa.hpp"

#define NAME "Comm_UM_Malloc"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

auto Comm_UM_Malloc = [] (benchmark::State &state,
  #if USE_NUMA
  const int numa_id
  #endif // USE_NUMA
  ) {

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

#if USE_NUMA
  numa_bind_node(numa_id);
#endif

  void *ptr = nullptr;
  
  for (auto _ : state) {
    auto start = std::chrono::system_clock::now();
    benchmark::DoNotOptimize(ptr = malloc(bytes));
    auto stop = std::chrono::system_clock::now();
    if (!ptr) {
      state.SkipWithError(NAME " failed to allocate");
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
  numa_bind_node(-1);
#endif
};

static void registerer() {
#if USE_NUMA
    for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
      ;
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_Malloc
#if USE_NUMA
        ,numa_id
#endif // USE_NUMA
        )->BYTE_ARGS()->UseManualTime();
#if USE_NUMA
    }
#endif // USE_NUMA
}

SCOPE_REGISTER_AFTER_INIT(registerer, NAME);


