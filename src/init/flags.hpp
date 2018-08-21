#ifndef COMM_SCOPE_SRC_INIT_FLAGS_HPP
#define COMM_SCOPE_SRC_INIT_FLAGS_HPP

#include "scope/utils/commandlineflags.hpp"

DECLARE_bool(version);

void init_flags(int argc, char **argv);

#endif
