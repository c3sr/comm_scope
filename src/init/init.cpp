#include "scope/init/init.hpp"

#include "numa.hpp"
#include "flags.hpp"

int comm_scope_init(int argc, char **argv) {
  (void) argc;
  (void) argv;

  init_flags(argc, argv);

  if (FLAG(version)) {
    std::cout << fmt::format("Comm|Scope") << std::endl;
  }

  init_numa();

  return 0;
}

SCOPE_INIT(comm_scope_init);
