# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED COMM_SCOPE_SRC_NUMA_MEMCPY_SUGAR_CMAKE_)
  return()
else()
  set(COMM_SCOPE_SRC_NUMA_MEMCPY_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    COMM_SCOPE_HEADERS
    args.hpp
)

sugar_files(
    COMM_SCOPE_SOURCES
    gpu_to_gpu_nopeer.cpp
    gpu_to_host.cpp
    gpu_to_pinned.cpp
    gpu_to_wc.cpp
    host_to_gpu.cpp
    host_to_pinned.cpp
    pinned_to_gpu.cpp
    wc_to_gpu.cpp
)

