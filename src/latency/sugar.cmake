# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED COMM_SCOPE_SRC_LATENCY_SUGAR_CMAKE_)
  return()
else()
  set(COMM_SCOPE_SRC_LATENCY_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    COMM_SCOPE_HEADERS
    args.hpp
    name.hpp
)

sugar_files(
    COMM_SCOPE_CUDA_SOURCES
    gpu_to_gpu.cu
    gpu_to_host.cu
    host_to_gpu.cu
)

