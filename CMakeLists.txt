option(SCOPE_USE_NUMA "Compile with support for libnuma" ON)
option(SCOPE_USE_OPENMP "Compile with support for OpenMP" ON)

# scope requires 3.18
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Comm|Scope LANGUAGES CXX VERSION 0.12.0)

# enable CUDA if desired and available
include(CheckLanguage)
if(SCOPE_USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(ERROR "No CUDA support")
  endif()
endif()

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

add_subdirectory(thirdparty/libscope)
add_subdirectory(src)
add_executable(comm_scope ${comm_SOURCES} ${comm_CUDA_SOURCES} src/main.cpp)

set_target_properties(comm_scope PROPERTIES ENABLE_EXPORTS ON) # better back trace

target_compile_options(comm_scope PRIVATE
$<$<COMPILE_LANGUAGE:CXX>:
    -Wall; 
    -Wextra;
    -Wcast-qual;
    -Wcast-align;
    -Wstrict-aliasing;
    -Wpointer-arith;
    -Winit-self;
    -Wshadow;
    -Wswitch-enum;
    -Wredundant-decls;
    -Wfloat-equal;
    -Wundef;
    -Wvla;
>
)

# causes lots of warnings in CUDA includes
# -Xcompiler=-Wundef;
# -Xcompiler=-Wredundant-decls;
# -Xcompiler=-fmax-errors=1;
target_compile_options(comm_scope PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo;
    -Xcompiler=-Wall; 
    -Xcompiler=-Wextra;
    -Xcompiler=-Wcast-qual;
    -Xcompiler=-Wcast-align;
    -Xcompiler=-Wstrict-aliasing;
    -Xcompiler=-Wpointer-arith;
    -Xcompiler=-Winit-self;
    -Xcompiler=-Wshadow;
    -Xcompiler=-Wswitch-enum;
    -Xcompiler=-Wfloat-equal;
    -Xcompiler=-Wvla;
>
  )

set_property(TARGET comm_scope PROPERTY CUDA_STANDARD 11)
target_compile_features(comm_scope PUBLIC cxx_std_11)
target_link_libraries(comm_scope scope::scope)

# libscope requires this for do_not_optimize
set_property(TARGET comm_scope PROPERTY CUDA_SEPARABLE_COMPILATION ON)

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
target_link_libraries(enable-turbo scope::scope)

add_executable(disable-turbo src/disable_turbo.cpp)
target_link_libraries(disable-turbo scope::scope)

add_executable(read-turbo src/read_turbo.cpp)
target_link_libraries(read-turbo scope::scope)

add_executable(set-maximum src/set_maximum.cpp)
target_link_libraries(set-maximum scope::scope)

add_executable(set-minimum src/set_minimum.cpp)
target_link_libraries(set-minimum scope::scope)
