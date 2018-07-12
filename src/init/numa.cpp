
#include <tuple>

#include "optional/optional.hpp"

#include "init/flags.hpp"
#include "init/init.hpp"
#include "init/logger.hpp"

#include "init/numa.hpp"

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
  LOG(debug, "set numa_exit_on_warm = 1");
  numa_exit_on_error = 1;
  LOG(debug, "set numa_exit_on_error = 1");
#endif // USE_NUMA == 1

  return false;
}
