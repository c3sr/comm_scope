#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyAsync_HostToPinned"

auto Comm_cudaMemcpyAsync_HostToPinned = [](benchmark::State &state,
                                            const int src_numa,
                                            const int dst_numa,
                                            const bool flush) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::bind_node(src_numa);
  void *src = aligned_alloc(page_size(), bytes);
  defer(free(src));
  std::memset(src, 0, bytes);

  numa::bind_node(dst_numa);
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
    if (flush) {
      numa::bind_node(src_numa);
      flush_all(src, bytes);
      numa::bind_node(dst_numa);
      flush_all(dst, bytes);
    }

    numa::bind_node(dst_numa);
    cudaEventRecord(start, NULL);
    const auto cuda_err =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToHost);
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
  numa::bind_node(-1);
};

static void registerer() {
  std::string name;

  const std::vector<MemorySpace> numaSpaces =
      scope::system::memory_spaces(MemorySpace::Kind::numa);

  for (auto src : numaSpaces) {
    for (auto dst : numaSpaces) {

      auto src_numa = src.numa_id();
      auto dst_numa = dst.numa_id();

      name = std::string(NAME) + "/" + std::to_string(src_numa) + "/" +
             std::to_string(dst_numa);
      benchmark::RegisterBenchmark(name.c_str(),
                                   Comm_cudaMemcpyAsync_HostToPinned, src_numa,
                                   dst_numa, false)
          ->SMALL_ARGS()
          ->UseManualTime();
      name = std::string(NAME) + "_flush/" + std::to_string(src_numa) + "/" +
             std::to_string(dst_numa);
      benchmark::RegisterBenchmark(name.c_str(),
                                   Comm_cudaMemcpyAsync_HostToPinned, src_numa,
                                   dst_numa, true)
          ->SMALL_ARGS()
          ->UseManualTime();
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
