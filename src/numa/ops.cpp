#include <benchmark/benchmark.h>

#include "ops.hpp"


void rd_8(char *ptr, const size_t count, const size_t stride)
{
    int64_t *dp = reinterpret_cast<int64_t*>(ptr);

    const size_t numElems = count / sizeof(int64_t);
    const size_t elemsPerStride = stride / sizeof(int64_t);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        benchmark::DoNotOptimize(dp[i]);
    }
}

void traverse(char *ptr, const size_t count)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < count; ++i) {
    benchmark::DoNotOptimize(ptr[i] = i);
  }
  benchmark::ClobberMemory();
}

void wr_8(char *ptr, const size_t count, const size_t stride)
{
    int64_t *dp = reinterpret_cast<int64_t*>(ptr);

    const size_t numElems = count / sizeof(int64_t);
    const size_t elemsPerStride = stride / sizeof(int64_t);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        benchmark::DoNotOptimize(dp[i] = 1);
    }
    benchmark::ClobberMemory();
}
