#if USE_NUMA && USE_OPENMP == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <numa.h>
#include <omp.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"
#include "init/numa.hpp"
#include "ops.hpp"
#include "utils/omp.hpp"

#define NAME "Comm/NUMA/WR"

static void Comm_NUMA_WR(benchmark::State &state) {

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const int threads  = state.range(0);
  const auto bytes   = 1ULL << static_cast<size_t>(state.range(1));
  const int src_numa = state.range(2);
  const int dst_numa = state.range(3);

  omp_set_num_threads(threads);
  if (threads != omp_get_max_threads()) {
    state.SkipWithError(NAME " unable to set OpenMP threads");
    return;
  }

  // Setup
  const long pageSize = sysconf(_SC_PAGESIZE);
  omp_numa_bind_node(dst_numa);
  char *ptr = static_cast<char *>(aligned_alloc(pageSize, bytes));
  std::memset(ptr, 0, bytes);

  for (auto _ : state) {
    state.PauseTiming();

    omp_numa_bind_node(dst_numa);

    std::memset(ptr, 0, bytes);
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();

    omp_numa_bind_node(src_numa);
    state.ResumeTiming();

    wr_8(ptr, bytes, 8);
  }

  omp_numa_bind_node(-1);

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
  state.counters["src_numa"] = src_numa;
  state.counters["dst_numa"] = dst_numa;

  free(ptr);
}

BENCHMARK(Comm_NUMA_WR)->Apply(ArgsThreadCount)->UseRealTime();

#endif // USE_NUMA == 1 && USE_OPENMP == 1
