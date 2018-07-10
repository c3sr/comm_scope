#if USE_NUMA == 1 && USE_OPENMP == 1

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <numa.h>
#include <omp.h>

#include "init/init.hpp"
#include "numa/args.hpp"
#include "ops.hpp"
#include "utils/utils.hpp"

#define NAME "NUMA/RD"

static void NUMA_RD(benchmark::State &state) {

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

  // Allocate ptr on src_numa
  const long pageSize = sysconf(_SC_PAGESIZE);
  omp_numa_bind_node(src_numa);
  char *ptr = static_cast<char *>(aligned_alloc(pageSize, bytes));

  // Make sure pages are allocated
  std::memset(ptr, 0, bytes);
  benchmark::DoNotOptimize(ptr);
  benchmark::ClobberMemory();

  // allocate some scratch space
  char *scratch = new char[64 * 1024 * 1024];
  defer(delete[] scratch);
  std::memset(ptr, 0, bytes);

  omp_numa_bind_node(dst_numa);
  for (auto _ : state) {
    state.PauseTiming();
    // invalidate dst cache
    omp_numa_bind_node(src_numa);
    std::memset(ptr, 0, bytes);
    benchmark::DoNotOptimize(ptr);
    benchmark::ClobberMemory();

    // Access from Device and Time
    // omp_numa_bind_node(dst_numa);
    state.ResumeTiming();

    rd_8(ptr, bytes, 8);
  }

  omp_numa_bind_node(-1);

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  free(ptr);
}

BENCHMARK(NUMA_RD)->Apply(ArgsThreadCountNumaNuma)->MinTime(0.1)->UseRealTime();

#endif // USE_NUMA == 1 && USE_OPENMP == 1