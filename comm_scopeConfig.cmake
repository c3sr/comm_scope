sugar_include("./src")

set(PREFIX ${CMAKE_CURRENT_LIST_DIR})

sugar_include("${PREFIX}/src")

set(comm_scope_BENCHMARK_SOURCES ${BENCHMARK_SOURCES})
set(comm_scope_BENCHMARK_HEADERS ${BENCHMARK_HEADERS})
set(comm_scope_INCLUDE_DIRS "${PREFIX}/src")
set(comm_scope_OPTIONAL_PACKAGES numa)
set(comm_scope_REQUIRED_PACKAGES "")