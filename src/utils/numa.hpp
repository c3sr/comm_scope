#pragma once

#if USE_NUMA == 1

#include <cassert>
#include <set>
#include <vector>

#include <numa.h>

#include "scope/init/logger.hpp"
#include "init/flags.hpp"

static inline void numa_bind_node(const int node) {

  if (-1 == node) {
    numa_bind(numa_all_nodes_ptr);
  } else if (node >= 0) {
    struct bitmask *nodemask = numa_allocate_nodemask();
    nodemask                 = numa_bitmask_setbit(nodemask, node);
    numa_bind(nodemask);
    numa_free_nodemask(nodemask);
  } else {
    LOG(critical, "expected node >= -1");
    exit(1);
  }
}

static inline size_t num_numa_nodes() {
  return FLAG(numa_ids).size();
}

#endif // USE_NUMA == 1

