#pragma once

#include "scope/scope.hpp"

/* use one thread from each warp to write a 0 to each stride
*/
template <bool NOOP = false>
__global__ void gpu_write(char *ptr, const size_t count, const size_t stride) {
  if (NOOP) {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx) {
    for (size_t i = wx * stride; i < count; i += numWarps * stride) {
      ptr[i] = 0;
    }
  }
}

extern std::condition_variable cv;
extern std::mutex m;
extern volatile bool ready;

inline void cpu_write(char *ptr, const size_t n, const size_t stride,
                      scope::time_point *start, scope::time_point *stop) {
  {
    std::unique_lock<std::mutex> lk(m);
    while (!ready)
      cv.wait(lk);
  }

  *start = scope::clock::now();
  for (size_t i = 0; i < n; i += stride) {
    benchmark::DoNotOptimize(ptr[i] = 0);
  }
  *stop = scope::clock::now();
}

struct Data {
  char *ptr;
  hipEvent_t start;
  hipEvent_t stop;
  size_t pageSize;
  bool error;
};

enum class Kind {
  GPUToGPU,
  GPUToHost,
  HostToGPU
};

template <Kind kind>
Data setup(benchmark::State &state,
                  const std::string &name,
                  const size_t bytes,
                  const int src_id,
                  const int dst_id);




template <Kind kind>
void prep_iteration(char *ptr, size_t bytes, int src_id, int dst_id);

