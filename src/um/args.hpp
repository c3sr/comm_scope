#pragma once

static void CustomArguments(benchmark::internal::Benchmark* b) {
  b->Args({0})->ArgName("log2(N)");
  for (int i = 12; i <= 33; ++i)
    for (auto j : {4096})
      b->Args({i})->ArgName("log2(N)");
}

#define BYTE_ARGS() Apply(CustomArguments)
