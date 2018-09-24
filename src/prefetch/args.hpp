#pragma once

#include "init/numa.hpp"
#include "utils/numa.hpp"

#define SMALL_ARGS() DenseRange(8, 32, 2)->ArgName("log2(N)")

