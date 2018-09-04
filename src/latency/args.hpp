#include "init/numa.hpp"
#include "utils/numa.hpp"

#include "scope/utils/utils.hpp"

#if USE_NUMA
inline static void ArgsCountNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (auto numa_id : numa_nodes()) {
    for (int gpu_id = 0; gpu_id < n; ++gpu_id) {
      for (int j = 4; j <= 34; ++j) {
        b->Args({j, numa_id, gpu_id});
      }
    }
  }
}
#endif

// Arguments of <log2(count), gpu_id>
inline static void ArgsCountGpu(benchmark::internal::Benchmark* b) {
  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }
  for (int gpu_id = 0; gpu_id < n; ++gpu_id) {
    for (int j = 4; j <= 34; ++j) {
      b->Args({j,gpu_id});
    }
  }
}


inline static void ArgsCountGpuGpuNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = gpu0 + 1; gpu1 < n; ++gpu1) {
      for (int j = 4; j <= 34; ++j) {
        b->Args({j, gpu0, gpu1});
      }
    }
  }
}