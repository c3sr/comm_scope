#pragma once

#include "scope/scope.hpp"

#include "../common/kernels.hpp"
#include "../common/kind.hpp"
#include "../common/unary_data.hpp"

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

template <Kind kind>
UnaryData setup(benchmark::State &state, const std::string &name,
                const size_t bytes, const int src_id, const int dst_id);

template <Kind kind>
void prep_iteration(char *ptr, size_t bytes, int src_id, int dst_id);
