#include <iostream>

#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/version.hpp"

#include "init/version.hpp"
#include "numa.hpp"
#include "config.hpp"

int comm_scope_init(int argc, char *const *argv) {

  if (FLAG(version)) {
    std::cout << version(SCOPE_PROJECT_NAME,
                         SCOPE_VERSION,
                         SCOPE_GIT_REFSPEC,
                         SCOPE_GIT_HASH,
                         SCOPE_GIT_LOCAL_CHANGES) << std::endl;

  }

  init_numa();

  return 0;
}

SCOPE_INIT(comm_scope_init);

