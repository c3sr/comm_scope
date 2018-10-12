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

#define NAME "Comm_UM_CudaMallocManaged"

#define OR_SKIP(stmt) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(NAME); \
    return; \
}

auto Comm_UM_CudaMallocManaged = [] (benchmark::State &state,
  #if USE_NUMA
  const int numa_id,
  #endif // USE_NUMA
  const int cuda_id) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const auto bytes   = 1ULL << static_cast<size_t>(state.range(0));

#if USE_NUMA
  numa_bind_node(numa_id);
#endif

  if (PRINT_IF_ERROR(utils::cuda_reset_device(cuda_id))) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA dst device");
    return;
  }

  char *ptr = nullptr;
  
  for (auto _ : state) {
    auto start = std::chrono::system_clock::now();
    cudaError_t err;
    err = cudaMallocManaged(&ptr, bytes);
    auto stop = std::chrono::system_clock::now();
    OR_SKIP(err);
    OR_SKIP(cudaFree(ptr));
    double seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
    state.SetIterationTime(seconds / 1e9);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
#if USE_NUMA
  state.counters["numa_id"] = numa_id;
#endif // USE_NUMA

#if USE_NUMA
  numa_bind_node(-1);
#endif
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
#if USE_NUMA
    for (auto numa_id : unique_numa_ids()) {
#endif // USE_NUMA
      std::string name = std::string(NAME)
#if USE_NUMA 
                       + "/" + std::to_string(numa_id) 
#endif // USE_NUMA
                       + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_CudaMallocManaged,
#if USE_NUMA
        numa_id,
#endif // USE_NUMA
        cuda_id)->BYTE_ARGS()->UseManualTime();
#if USE_NUMA
    }
#endif // USE_NUMA
  }
}

SCOPE_REGISTER_AFTER_INIT(registerer, NAME);


