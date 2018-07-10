#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include "init/init.hpp"
#include "utils/utils.hpp"


static void EXAMPLE1(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(EXAMPLE1);

// Define another benchmark
static void EXAMPLE2(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(EXAMPLE2);


