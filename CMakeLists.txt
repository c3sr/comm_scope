option(USE_NUMA "Compile with support for libnuma" ON)
option(USE_NVTX "Add NVTX ranges to some benchmarks" ON)
option(USE_OpenMP "Use OpenMP for some benchmarks" ON)

# sysbench requires 3.17
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(Comm|Scope LANGUAGES CXX CUDA VERSION 0.9.0)

# Set a default build type if none was specified
set(default_build_type "Release")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure sysbench to match
# FORCE to override variable if it's already cached
set(SYSBENCH_USE_CUDA ON CACHE BOOL "use CUDA" FORCE)
set(SYSBENCH_USE_NUMA ${USE_NUMA} CACHE BOOL "text" FORCE)
set(SYSBENCH_USE_OPENMP ${USE_OPENMP} CACHE BOOL "" FORCE)
set(SYSBENCH_USE_NVTX ${USE_NVTX} CACHE BOOL "" FORCE)

add_subdirectory(thirdparty/sysbench)
add_subdirectory(src)
add_executable(comm_scope ${comm_SOURCES} ${comm_CUDA_SOURCES})

target_compile_options(comm_scope PRIVATE
    -Wall 
    -Wextra 
    -Wcast-qual 
    -Wcast-align 
    -Wstrict-aliasing 
    -Wpointer-arith 
    -Winit-self 
    -Wshadow 
    -Wswitch-enum 
    -Wredundant-decls 
    -Wfloat-equal 
    -Wundef 
    -Wvla
    -fmax-errors=1 
    -Wfatal-errors
  )

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

add_executable(enable-turbo src/enable_turbo.cpp)
target_link_libraries(enable-turbo sysbench::sysbench)

add_executable(set-maximum src/set_maximum.cpp)
target_link_libraries(set-maximum sysbench::sysbench)

add_executable(set-minimum src/set_minimum.cpp)
target_link_libraries(set-minimum sysbench::sysbench)