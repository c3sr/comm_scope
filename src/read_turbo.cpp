#include <iostream>

#include "scope/scope.hpp"

int main(int argc, char **argv) {
  using namespace turbo;
  State state;
  Result res = get_state(&state);
  if (Result::SUCCESS != res) {
    std::cout << "read turbo: " << turbo::get_string(turbo::enable()) << "\n";
    return 1;
  }

  std::cout << "read turbo: " << state.enabled << " ("
            << turbo::get_string(state.method) << ")\n";
  return 0;
}