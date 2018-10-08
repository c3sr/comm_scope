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

sugar_include(coherence)
sugar_include(init)
sugar_include(latency)
sugar_include(mapping)
sugar_include(memcpy)
sugar_include(memcpy-duplex)
sugar_include(numa-memcpy)
sugar_include(numamemcpy-duplex)
sugar_include(prefetch)
sugar_include(utils)

