#pragma once

#include "scope/do_not_optimize.hpp"
#include "scope/clobber.hpp"

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