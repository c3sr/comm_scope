# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_EXAMPLE2_SUGAR_CMAKE_)
  return()
else()
  set(SRC_EXAMPLE2_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    SCOPE_COMM_SOURCES
    example.cpp
)

