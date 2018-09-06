#include <iostream>

#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/version.hpp"
#include "scope/utils/commandlineflags.hpp"

#include "numa.hpp"
#include "config.hpp"
#include "flags.hpp"

int comm_scope_init(int argc, char *const *argv) {

  if (FLAG(version)) {
    std::cout << version(SCOPE_PROJECT_NAME,
                         SCOPE_VERSION,
                         SCOPE_GIT_REFSPEC,
                         SCOPE_GIT_HASH,
                         SCOPE_GIT_LOCAL_CHANGES) << std::endl;

  }

  for (int i = 1; i < argc; ++i) {
    utils::ParseVecInt32Flag(argv[i], "numa_ids", &FLAG(numa_ids));
  }
  for (const auto &e : FLAG(numa_ids)) {
    LOG(debug, "User requested NUMA node " + std::to_string(e));
  }

  if (!init_numa()) {
    LOG(critical, "Error setting up NUMA");
    return -1;
  }

  return 0;
}

SCOPE_INIT(comm_scope_init);

