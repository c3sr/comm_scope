#ifndef UTILS_NUMA_HPP
#define UTILS_NUMA_HPP

#if USE_NUMA == 1

#include "init/logger.hpp"
#include <cassert>
#include <numa.h>
#include <set>
#include <vector>

inline std::vector<int> numa_nodes() {
  std::set<int> nodes;
  for (int i = 0; i < numa_num_configured_cpus(); ++i) {
    nodes.insert(numa_node_of_cpu(i));
  }
  assert(nodes.size() >= 1);
  std::vector<int> nodes2;
  for (const auto &i : nodes) {
    nodes2.push_back(i);
  }
  return nodes2;
}

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

#endif // USE_NUMA == 1

#endif // UTILS_NUMA_HPP