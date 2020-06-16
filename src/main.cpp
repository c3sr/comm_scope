#include "sysbench/sysbench.hpp"

int main(int argc, char **argv) {
  sysbench::initialize(&argc, argv);
  sysbench::run();
}