#include <iostream>

#include "scope/scope.hpp"

#include "config.hpp"

int main(int argc, char **argv) {

  std::cout << "Comm|Scope " << COMM_SCOPE_VERSION_MAJOR << "."
            << COMM_SCOPE_VERSION_MINOR << "." << COMM_SCOPE_VERSION_PATCH
            << " " << COMM_SCOPE_GIT_REFSPEC << "@" << SCOPE_GIT_HASH << "\n";
  std::cout << "  scope " << SCOPE_VERSION_MAJOR << "." << SCOPE_VERSION_MINOR
            << "." << SCOPE_VERSION_PATCH << " " << SCOPE_GIT_REFSPEC << "@"
            << SCOPE_GIT_HASH << "\n";
  std::cout << "    spdlog " << SPDLOG_VER_MAJOR << "." << SPDLOG_VER_MINOR
            << "." << SPDLOG_VER_PATCH << "\n";
  std::cout << "    benchmark 1.5.1\n";

  scope::initialize(&argc, argv);
  scope::run();
  scope::finalize();
}