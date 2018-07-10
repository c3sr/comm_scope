
#define SMALL_ARGS() DenseRange(8, 31, 2)->ArgName("log2(N)")

#define ALL_ARGS() DenseRange(1, 31, 1)->ArgName("log2(N)")

inline
static void ArgsCountGpuGpuNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = 0; gpu1 < n; ++gpu1) {
      if (gpu0 != gpu1) {
        for (int j = 8; j <= 31; ++j) {
          b->Args({j, gpu0, gpu1});
        }
      }
    }
  }
}

inline
static void ArgsCountGpuGpuPeerNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = 0; gpu1 < n; ++gpu1) {

      int peer = 0;
      cudaError_t err = cudaDeviceCanAccessPeer(&peer, gpu1, gpu0);
      if (err != cudaSuccess || 0 == peer) {
        continue;
      }
      err = cudaDeviceCanAccessPeer(&peer, gpu0, gpu1);
      if (err != cudaSuccess || 0 == peer) {
        continue;
      }

      if (gpu0 != gpu1) {
        for (int j = 8; j <= 31; ++j) {
          b->Args({j, gpu0, gpu1});
        }
      }
    }
  }
}