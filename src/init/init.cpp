#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"

#include "init/version.hpp"
#include "numa.hpp"

int comm_scope_init(int argc, char *const *argv) {

  if (FLAG(version)) {
    std::cout << fmt::format("Comm|Scope ") << comm_scope::version() << std::endl;
  }

  init_numa();

  return 0;
}

SCOPE_INIT(comm_scope_init);
