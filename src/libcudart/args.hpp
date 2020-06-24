#pragma once

#define ALLOC_ARGS() Arg(0)->DenseRange(12,33,1)->ArgName("log2(N)")

#define GRAPH_ARGS() Arg(0)->Arg(1)->DenseRange(2,20,2)