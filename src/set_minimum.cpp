#include <iostream>

#include "sysbench/sysbench.hpp"

int main(int argc, char **argv) {
  std::cout << "set minimum: " << governor::get_string(governor::set_state_minimum()) << "\n";
}