#if USE_NUMA == 1

#include <cassert>

#include <cuda_runtime.h>
#include <numa.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/utils/page_size.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "init/numa.hpp"
#include "utils/numa.hpp"

#define NAME "Comm_NUMAMemcpy_HostToPinned"

auto Comm_NUMAMemcpy_HostToPinned = [](benchmark::State &state, const int src_numa, const int dst_numa) {

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

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));

  numa_bind_node(src_numa);
  void *src = aligned_alloc(page_size(), bytes);
  defer(free(src));
  std::memset(src, 0, bytes);

  numa_bind_node(dst_numa);
  void *dst = aligned_alloc(page_size(), bytes);
  std::memset(dst, 0, bytes);
  if (PRINT_IF_ERROR(cudaHostRegister(dst, bytes, cudaHostRegisterPortable))) {
    state.SkipWithError(NAME " failed to register allocations");
    return;
  }
  defer(cudaHostUnregister(dst));
  defer(free(dst));

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
  state.counters["bytes"] = bytes;
  state.counters["src_numa"] = src_numa;
  state.counters["dst_numa"] = dst_numa;

  // reset to run on any node
  numa_bind_node(-1);
};

static void registerer() {
  for (auto src_numa : unique_numa_ids()) {
    for (auto dst_numa : unique_numa_ids()) {
      std::string name = std::string(NAME) + "/" + std::to_string(src_numa) + "/" + std::to_string(dst_numa);
      benchmark::RegisterBenchmark(name.c_str(), Comm_NUMAMemcpy_HostToPinned, src_numa, dst_numa)->SMALL_ARGS()->UseManualTime();
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // USE_NUMA == 1
