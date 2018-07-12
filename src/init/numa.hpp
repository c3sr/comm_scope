#ifndef INIT_NUMA_HPP
#define INIT_NUMA_HPP

#if USE_NUMA == 1
#include <numa.h>
#endif // USE_NUMA == 1

extern bool has_numa;
bool init_numa();

#endif