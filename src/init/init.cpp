#include <iostream>

#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/version.hpp"
#include "scope/utils/commandlineflags.hpp"

#include "numa.hpp"
#include "config.hpp"
#include "flags.hpp"

void comm_before_init() {
  RegisterVersionString(
    version(SCOPE_PROJECT_NAME,
                         SCOPE_VERSION,
                         SCOPE_GIT_REFSPEC,
                         SCOPE_GIT_HASH,
                         SCOPE_GIT_LOCAL_CHANGES)
  );

  RegisterOpt(  clara::Opt(FLAG(numa_ids), "id")["-n"]["--numa"]("add numa device id (Comm|Scope)")  );
}

int comm_scope_init() {

  for (const auto &e : FLAG(numa_ids)) {
    LOG(debug, "User requested NUMA node " + std::to_string(e));
  }

  if (!init_numa()) {
    LOG(critical, "Error setting up NUMA");
    return -1;
  }

  return 0;
}

SCOPE_REGISTER_BEFORE_INIT(comm_before_init);
SCOPE_REGISTER_INIT(comm_scope_init);
