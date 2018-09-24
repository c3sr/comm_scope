#if USE_NUMA == 1

#include <cassert>

#include <cuda_runtime.h>
#include <numa.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"

#include "args.hpp"
#include "init/flags.hpp"
#include "init/numa.hpp"
#include "utils/numa.hpp"

#define NAME "Comm_NUMAMemcpy_PinnedToGPU"

auto Comm_NUMAMemcpy_PinnedToGPU = [](benchmark::State &state, const int numa_id, const int cuda_id) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const auto bytes  = 1ULL << static_cast<size_t>(state.range(0));


  numa_bind_node(numa_id);
  if (PRINT_IF_ERROR(utils::cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }

  void *src = aligned_alloc(page_size(), bytes);
  char *dst = nullptr;

  std::memset(src, 0, bytes);
  if (PRINT_IF_ERROR(cudaHostRegister(src, bytes, cudaHostRegisterPortable))) {
    state.SkipWithError(NAME " failed to register allocation");
    return;
  }
  defer(cudaHostUnregister(src));
  defer(free(src));

  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }

  if (PRINT_IF_ERROR(cudaMalloc(&dst, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(dst));

  if (PRINT_IF_ERROR(cudaMemset(dst, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);
    const auto cuda_err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    state.PauseTiming();

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
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;

  // reset to run on any node
  numa_bind_node(-1);
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : unique_numa_ids()) {
      std::string name = std::string(NAME) + "/" + std::to_string(numa_id) + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_NUMAMemcpy_PinnedToGPU, numa_id, cuda_id)->SMALL_ARGS()->UseManualTime();
    }
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

#endif // USE_NUMA == 1
