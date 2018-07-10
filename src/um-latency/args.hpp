inline
static void ArgsCountGpuGpuNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = gpu0+1; gpu1 < n; ++gpu1) {
      for (int j = 4; j <= 34; ++j) {
        b->Args({j, gpu0, gpu1});
      }
    }
  }
}