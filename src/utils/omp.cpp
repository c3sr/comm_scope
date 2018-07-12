#if USE_OPENMP == 1
#if USE_NUMA == 1

#include "omp.hpp"
#include "numa.hpp"

void omp_numa_bind_node(const int numa_id) {
  numa_bind_node(numa_id);
#pragma omp parallel
  { numa_bind_node(numa_id); }
}

#endif // USE_NUMA == 1
#endif // USE_OPENMP == 1