
#include <tuple>

#include "optional/optional.hpp"

#include "scope/utils/commandlineflags.hpp"
#include "scope/init/init.hpp"
#include "scope/init/logger.hpp"

#include "init/numa.hpp"
#include "init/flags.hpp"

bool has_numa = false;

bool init_numa() {

#if USE_NUMA == 1
  int ret = numa_available();
  if (-1 == ret) {
    LOG(critical, "NUMA not available");
    exit(1);
  } else {
    has_numa = true;
  }

  numa_set_strict(1);
  LOG(debug, "set numa_set_strict(1)");
  numa_set_bind_policy(1);
  LOG(debug, "set numa_set_bind_policy(1)");

  numa_exit_on_warn = 1;
  LOG(debug, "set numa_exit_on_warn = 1");
  numa_exit_on_error = 1;
  LOG(debug, "set numa_exit_on_error = 1");

  // discover available nodes
  std::set<int> available_nodes;
  for (int i = 0; i < numa_num_configured_cpus(); ++i) {
    available_nodes.insert(numa_node_of_cpu(i));
  }

  // set numa ids to available nodes
  if (FLAG(numa_ids).empty()) {
    for (const auto &id : available_nodes) {
      FLAG(numa_ids).push_back(id);
    }
  } else { // check to make sure requested numa ids are valid
    for (const auto &id : FLAG(numa_ids)) {
      if (0 == available_nodes.count(id)) {
        LOG(critical, fmt::format("user-requested NUMA node id {} is not available.", id));
        for (const auto &avail : available_nodes) {
           LOG(critical, fmt::format("NUMA node {} is available.", avail));
        }
        return false;
      }
    }
  }

#endif // USE_NUMA == 1

  return true;
}
