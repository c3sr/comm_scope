#pragma once

inline
static void ArgsThreadCount(benchmark::internal::Benchmark* b) {

  for (int t = 1; t <= 8; t *= 2) { // threads
    for (int j = 14; j <= 36; ++j) { // log2(bytes)
      b->Args({t, j});
    }
  }
}
