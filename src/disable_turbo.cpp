#include <iostream>

#include "scope/scope.hpp"

int main(int argc, char **argv) {
  using namespace turbo;
  std::cout << "disable turbo: " << get_string(disable()) << "\n";
}