#include "scope/init/init.hpp"

#include "init/numa.hpp"

int comm_scope_init(int argc, char **argv) {
  (void) argc;
  (void) argv;

  init_numa();

  return 0;
}

SCOPE_INIT(comm_scope_init);
