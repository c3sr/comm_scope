#include <cstdint>
#include <cstdlib>

inline static size_t popcount(uint64_t u) {
  return __builtin_popcount(u);
}

inline static bool is_set(uint64_t bits, size_t i) {
  return (uint64_t(1) << i) & bits;
}

inline static void ArgsCountGpuGpuNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = gpu0 + 1; gpu1 < n; ++gpu1) {
      for (int j = 8; j < 31; ++j) {
        b->Args({j, gpu0, gpu1});
      }
    }
  }

}

/*
inline static void ArgsMultiGPU(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  // maximum bit pattern where each bit represents a GPU
  assert(n < 64);
  const uint64_t gpu_bits_max = uint64_t(1) << n;

  // number of gpus participating in this benchmark
  for (int num_gpus = 0; num_gpus <= n; ++num_gpus) {

    // all bit patterns representing GPUs that could participate
    for (uint64_t gpu_bits = 0; gpu_bits < gpu_bits_max; ++gpu_bits) {

      // if there are the right number of GPUs, include in arguments
      if (num_gpus == popcount(gpu_bits)) {
        for (int j = 8; j <= 31; ++j) {

          // Build up a meaningful name instead of gpu_bits
          std::string arg_name;
          for (int i = 0; i < 64; ++i) {
            if (is_set(gpu_bits, i)) {
              arg_name += std::to_string(i) + "-";
            }
          }
          if (arg_name.size() > 0) {
            // remove trailing -
            arg_name.resize(arg_name.size() - 1);
          }

          b->Args({j, num_gpus, gpu_bits})->ArgName(arg_name);
        }
      }
    }
  }
}
*/
