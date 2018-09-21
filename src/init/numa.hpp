#pragma once

#include <vector>

#if USE_NUMA == 1
#include <numa.h>
#endif // USE_NUMA == 1

extern bool has_numa;
bool init_numa();

const std::vector<int> &unique_numa_ids();
