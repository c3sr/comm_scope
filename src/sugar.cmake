# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED COMM_SCOPE_SRC_SUGAR_CMAKE_)
  return()
else()
  set(COMM_SCOPE_SRC_SUGAR_CMAKE_ 1)
endif()

include(sugar_include)

sugar_include(memcpy)
sugar_include(numa)
sugar_include(numamemcpy)
sugar_include(numaum-coherence)
sugar_include(numaum-latency)
sugar_include(numaum-prefetch)
sugar_include(um-coherence)
sugar_include(um-latency)
sugar_include(um-prefetch)

