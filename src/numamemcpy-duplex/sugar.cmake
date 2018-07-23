# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED COMM_SCOPE_SRC_NUMAMEMCPY_DUPLEX_SUGAR_CMAKE_)
  return()
else()
  set(COMM_SCOPE_SRC_NUMAMEMCPY_DUPLEX_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    comm_scope_HEADERS
    args.hpp
)

sugar_files(
    comm_scope_SOURCES
    gpu_to_host.cpp
    gpu_to_pinned.cpp
    host_to_gpu.cpp
    pinned_to_gpu.cpp
)

