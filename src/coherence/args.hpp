#include "init/numa.hpp"
#include "utils/numa.hpp"

#include "scope/utils/utils.hpp"

inline static void ArgsCountNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (auto numa_id : numa_nodes()) {
    for (int gpu_id = 0; gpu_id < n; ++gpu_id) {
      for (int j = 8; j <= 33; ++j) {
        b->Args({j, numa_id, gpu_id});
      }
    }
  }
}

inline static void ArgsThreadsNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int j = 0; j <= 11; ++j) {
    for (auto numa_id : numa_nodes()) {
      for (int gpu_id = 0; gpu_id < n; ++gpu_id) {
        b->Args({j, numa_id, gpu_id});
      }
    }
  }
}

inline static void ArgsThreadsCountNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int t = 1; t <= 8; t *= 2) {
    for (int j = 8; j <= 31; ++j) {
      for (auto numa_id : numa_nodes()) {
        for (int gpu_id = 0; gpu_id < n; ++gpu_id) {
          b->Args({t, j, numa_id, gpu_id});
        }
      }
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
    for (int gpu1 = 0; gpu1 < n; ++gpu1) {
      if (gpu0 != gpu1) {
        for (int j = 8; j <= 33; ++j) {
          b->Args({j, gpu0, gpu1});
        }
      }
    }
  }
}
