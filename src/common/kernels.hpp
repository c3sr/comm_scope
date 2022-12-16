#pragma once

#include "scope/do_not_optimize.hpp"
#include "scope/clobber.hpp"


template <typename write_t>
__global__ void gpu_write(void *ptr, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(write_t);

  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    reinterpret_cast<write_t*>(ptr)[i] = i;
  }
}

template <typename write_t>
__global__ void gpu_write_zero(void *ptr, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(write_t);

  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    reinterpret_cast<write_t*>(ptr)[i] = 0;
  }
}

/*
The compiler will try to optimize unused reads away.
`asm(...)` should prevent the load from being optimized out of the PTX.
However, the JIT can still tell to get rid of it.
The `flag` parameter prevents the JIT from removing the load, since it might be
used at runtime. We pass `flag = false` so the store is not executed. We still
need `asm(...)` however, to prevent the load from being lowered into the
conditional and skipped.
*/
template <typename read_t>
__global__ void gpu_read(const void *ptr, void *flag, const size_t bytes) {
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_elems = bytes / sizeof(read_t);

  // #pragma unroll(1)
  for (size_t i = gx; i < num_elems; i += gridDim.x * blockDim.x) {
    read_t t;
    scope::do_not_optimize(t = reinterpret_cast<const read_t*>(ptr)[i]);
    if (flag) {
      *reinterpret_cast<read_t*>(flag) = t;
    }
  }
}