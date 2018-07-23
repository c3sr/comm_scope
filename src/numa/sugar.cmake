# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED COMM_SCOPE_SRC_NUMA_SUGAR_CMAKE_)
  return()
else()
  set(COMM_SCOPE_SRC_NUMA_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    comm_HEADERS
    args.hpp
    ops.hpp
)

sugar_files(
    comm_SOURCES
    ops.cpp
    rd.cpp
    wr.cpp
)

