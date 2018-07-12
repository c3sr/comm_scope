#define SMALL_ARGS() DenseRange(8, 31, 1)->ArgName("log2(N)")

inline
static void ArgsCountNumaNuma(benchmark::internal::Benchmark* b) {

  for (auto src_numa : numa_nodes()) {
    for (auto dst_numa : numa_nodes()) {
      for (int j = 8; j <= 33; ++j) {
        b->Args({j, src_numa, dst_numa});
      }
    }
  }
}

inline
static void ArgsCountNumaGpu(benchmark::internal::Benchmark* b) {

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

inline
static void ArgsCountNumaGpuGpuNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (auto numa_id : numa_nodes()) {
    for (int gpu0 = 0; gpu0 < n; ++gpu0) {
      for (int gpu1 = 0; gpu1 < n; ++gpu1) {
        if (gpu0 != gpu1) {
          for (int j = 8; j <= 33; ++j) {
            b->Args({j, numa_id, gpu0, gpu1});
          }
        }
      }
    }
  }
}