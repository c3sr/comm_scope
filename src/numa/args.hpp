inline
static void ArgsCountNumaNuma(benchmark::internal::Benchmark* b) {

  for (int j = 12; j <= 32; ++j) { // log2(bytes)
      for (auto src_numa : numa_nodes()) {
        for (auto dst_numa : numa_nodes()) {
            b->Args({j, src_numa, dst_numa});
        }
      }
    }
}

inline
static void ArgsThreadCountNumaNuma(benchmark::internal::Benchmark* b) {

  for (int t = 1; t <= 8; t *= 2) { // threads
    for (int j = 14; j <= 36; ++j) { // log2(bytes)
      for (auto src_numa : numa_nodes()) {
        for (auto dst_numa : numa_nodes()) {
            b->Args({t, j, src_numa, dst_numa});
        }
      }
    }
  }
}
