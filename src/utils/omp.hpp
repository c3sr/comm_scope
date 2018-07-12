#ifndef OMP_HPP
#define OMP_HPP

#if USE_OPENMP == 1
#if USE_NUMA == 1
#include <omp.h>

void omp_numa_bind_node(const int numa_id);

#endif // USE_NUMA == 1
#endif // USE_OMP == 1

#endif