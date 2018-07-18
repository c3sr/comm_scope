#include <cstdint>
#include <cstdlib>


inline static void ArgsCountNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (auto numa : numa_nodes()) {
    for (int gpu = 0; gpu < n; ++gpu) {
      for (int j = 8; j <= 32; ++j) {
        b->Args({j, numa, gpu});
      }
    }
  }
}
