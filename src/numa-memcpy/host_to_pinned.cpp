#if USE_NUMA == 1

#include <cassert>

#include <cuda_runtime.h>
#include <numa.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "init/numa.hpp"
#include "utils/numa.hpp"

#define NAME "Comm/NUMAMemcpy/HostToPinned"

static void Comm_NUMAMemcpy_HostToPinned(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  if (num_numa_nodes() < 2) {
    state.SkipWithError(NAME " requires two NUMA nodes");
    return;
  }

  const int src_numa = FLAG(numa_ids)[0];
  const int dst_numa = FLAG(numa_ids)[1];

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));


  numa_bind_node(src_numa);
  char *src = new char[bytes];
  defer(delete[] src);
  std::memset(src, 0, bytes);

  numa_bind_node(dst_numa);
  char *dst = new char[bytes];
  std::memset(dst, 0, bytes);
  if (PRINT_IF_ERROR(cudaHostRegister(dst, bytes, cudaHostRegisterPortable))) {
    state.SkipWithError(NAME " failed to register allocations");
    return;
  }
  defer(cudaHostUnregister(dst));
  defer(delete[] dst);

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    // Invalidate dst cache (if different from src)
    numa_bind_node(src_numa);
    std::memset(dst, 0, bytes);
    benchmark::DoNotOptimize(dst);
    benchmark::ClobberMemory();

    numa_bind_node(dst_numa);
    cudaEventRecord(start, NULL);
    const auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToHost);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError(NAME " failed to perform memcpy");
      break;
    }
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(NAME " failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  // reset to run on any node
  numa_bind_node(-1);
}

BENCHMARK(Comm_NUMAMemcpy_HostToPinned)->SMALL_ARGS()->UseManualTime();

#endif // USE_NUMA == 1
