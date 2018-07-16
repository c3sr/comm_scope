#include "scope/init/init.hpp"

#include "init/numa.hpp"

void myinit(int argc, char **argv) {
  (void) argc;
  (void) argv;

  init_numa();
}

INIT(myinit);
