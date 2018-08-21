#include "flags.hpp"

DEFINE_bool(version, false, "Show version message.");

static void parse(int* argc, char** argv) {
  using namespace utils;
  for (int i = 1; i < *argc; ++i) {
    ParseBoolFlag(argv[i], "version", &FLAG(version));
  }

}

void init_flags(int argc, char** argv) {
  parse(&argc, argv);

  return;
}