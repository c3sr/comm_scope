option(ENABLE_COMM "Include Comm|Scope (github.com/c3sr/comm_scope)" ON)
if (NOT ENABLE_COMM)
  return()
endif()


cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(Comm|Scope LANGUAGES CXX CUDA VERSION 0.2.0)
include(ScopeStatus)

sugar_include("src")

scope_add_library(comm_scope OBJECT ${comm_SOURCES} ${comm_CUDA_SOURCES} )
target_include_scope_directories(comm_scope)
target_include_directories(comm_scope PRIVATE
  ${PROJECT_BINARY_DIR}/src
  src
)
target_link_scope_libraries(comm_scope)

target_compile_features(comm_scope PUBLIC cxx_std_11)
set_property(TARGET comm_scope PROPERTY CUDA_STANDARD 11)

## NUMA
find_package(numa)
if(numa_FOUND)
    target_compile_definitions(comm_scope PRIVATE -DUSE_NUMA=1)
    target_link_libraries(comm_scope PUBLIC numa)
    target_include_directories(comm_scope SYSTEM PRIVATE ${numa_INCLUDE_DIR})
else()
target_compile_definitions(comm_scope PRIVATE -DUSE_NUMA=0)
  message(WARNING "didn't find NUMA, some benchmarks will be unavailable.")
endif(numa_FOUND)

## OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
  target_link_libraries(comm_scope PUBLIC ${OpenMP_CXX_LIBRARIES})
  if (OpenMP_C_FLAGS)
    string(REPLACE ";" " " OpenMP_C_FLAGS ${OpenMP_C_FLAGS})
  endif()
  if (OpenMP_CXX_FLAGS)
    string(REPLACE ";" " " OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  foreach(flag IN ITEMS ${OpenMP_CXX_FLAGS})
  LIST(APPEND CMAKE_CUDA_FLAGS
    -Xcompiler
    ${flag}
  )
  endforeach(flag)

  add_definitions(-DUSE_OPENMP=1)
else(OpenMP_FOUND)
  add_definitions(-DUSE_OPENMP=0)
  message(WARNING "didn't find OpenMP, some benchmarks will be unavailable.")
endif(OpenMP_FOUND)

# CUDA
target_compile_definitions(comm_scope PRIVATE -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
target_compile_definitions(comm_scope PRIVATE -DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})
scope_status("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
scope_status("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Get the git version
git_get_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
scope_status(GIT_REFSPEC=${GIT_REFSPEC})
scope_status(GIT_HASH=${GIT_HASH})
scope_status(GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

# Create the version file
scope_status("${PROJECT_SOURCE_DIR}/.../config.hpp.in -> ${PROJECT_BINARY_DIR}/.../config.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/src/config.hpp.in"
    "${PROJECT_BINARY_DIR}/src/config.hpp"
)

