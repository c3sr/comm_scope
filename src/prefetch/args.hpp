#pragma once

#include "init/numa.hpp"
#include "utils/numa.hpp"

#define SMALL_ARGS() DenseRange(8, 31, 2)->ArgName("log2(N)")

