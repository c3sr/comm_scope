#include "sysbench/sysbench.hpp"

#include "args.hpp"

#define NAME "Comm_UM_CudaMallocManaged"

#define OR_SKIP(stmt)                                                          \
  if (PRINT_IF_ERROR(stmt)) {                                                  \
    state.SkipWithError(NAME);                                                 \
    return;                                                                    \
  }

auto Comm_UM_CudaMallocManaged = [](benchmark::State &state,
                                    const int numa_id,
                                    const int cuda_id) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  numa::bind_node(numa_id);

  if (PRINT_IF_ERROR(cuda_reset_device(cuda_id))) {
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
    double seconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
            .count();
    state.SetIterationTime(seconds / 1e9);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["cuda_id"] = cuda_id;
  state.counters["numa_id"] = numa_id;

  numa::bind_node(-1);
};

static void registerer() {
  for (auto cuda_id : unique_cuda_device_ids()) {
    for (auto numa_id : numa::ids()) {
      std::string name = std::string(NAME)
                         + "/" + std::to_string(numa_id)
                         + "/" + std::to_string(cuda_id);
      benchmark::RegisterBenchmark(name.c_str(), Comm_UM_CudaMallocManaged,
                                   numa_id,
                                   cuda_id)
          ->BYTE_ARGS()
          ->UseManualTime();
    }
  }
}

SYSBENCH_AFTER_INIT(registerer, NAME);
