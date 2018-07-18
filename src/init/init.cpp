#include "scope/init/init.hpp"

#include "init/numa.hpp"

void comm_scope_init(int argc, char **argv) {
  (void) argc;
  (void) argv;

  init_numa();
}

SCOPE_INIT(comm_scope_init);
