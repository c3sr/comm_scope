#include <iostream>

#include "scope/scope.hpp"

int main(int argc, char **argv) {
  std::cout << "set maximum: "
            << governor::get_string(governor::set_state_maximum()) << "\n";
}