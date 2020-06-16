option(USE_NUMA "Compile with support for libnuma" ON)
option(USE_NVTX "Add NVTX ranges to some benchmarks" ON)
option(USE_OpenMP "Use OpenMP for some benchmarks" ON)

# sysbench requires 3.17
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(Comm|Scope LANGUAGES CXX CUDA VERSION 0.9.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(thirdparty/sysbench)

add_subdirectory(src)

add_executable(comm_scope ${comm_SOURCES} ${comm_CUDA_SOURCES})


set_property(TARGET comm_scope PROPERTY CUDA_STANDARD 11)
target_compile_features(comm_scope PUBLIC cxx_std_11)
target_link_libraries(comm_scope sysbench::sysbench)

# Get the git version
git_get_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

# Create the version file and find it
message(STATUS "${PROJECT_SOURCE_DIR}/.../config.hpp.in -> ${PROJECT_BINARY_DIR}/.../config.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/src/config.hpp.in"
    "${PROJECT_BINARY_DIR}/src/config.hpp"
)
target_include_directories(comm_scope PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/src)

