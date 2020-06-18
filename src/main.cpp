#include "sysbench/sysbench.hpp"

#include "config.hpp"

int main(int argc, char **argv) {

  std::cout << "Comm|Scope " << SCOPE_VERSION_MAJOR << "."
            << SCOPE_VERSION_MINOR << "." << SCOPE_VERSION_PATCH << " " << SCOPE_GIT_REFSPEC << "@" << SCOPE_GIT_HASH << "\n";
  std::cout << "  sysbench " << SYSBENCH_VERSION_MAJOR << "."
            << SYSBENCH_VERSION_MINOR << "." << SYSBENCH_VERSION_PATCH << " "
            << SYSBENCH_GIT_REFSPEC << "@" << SYSBENCH_GIT_HASH << "\n";
  std::cout << "    spdlog " << SPDLOG_VER_MAJOR << "." << SPDLOG_VER_MINOR
            << "." << SPDLOG_VER_PATCH << "\n";
  std::cout << "    benchmark 1.5.1\n";

  sysbench::initialize(&argc, argv);
  sysbench::run();
  sysbench::finalize();
}