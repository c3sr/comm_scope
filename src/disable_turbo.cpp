#include <iostream>

#include "sysbench/sysbench.hpp"

int main(int argc, char **argv) {
  using namespace turbo;
  std::cout << "disable turbo: " << get_string(disable()) << "\n";
}