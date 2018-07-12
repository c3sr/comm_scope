# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED COMM_SCOPE_SRC_NUMAUM_COHERENCE_SUGAR_CMAKE_)
  return()
else()
  set(COMM_SCOPE_SRC_NUMAUM_COHERENCE_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    gpu_threads.cu
    gpu_to_host.cu
    gpu_to_host_threads.cu
    host_to_gpu.cu
)

