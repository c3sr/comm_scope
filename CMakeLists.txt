cmake_minimum_required(VERSION 3.8)

# if(MSVC)
#     target_compile_options(/W3 /WX)
# else()
#     target_compile_options(-Wall -Werror)
# endif()


sugar_include("src")

add_library(comm_scope ${COMM_SCOPE_SOURCES} ${COMM_SCOPE_CUDA_SOURCES} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
target_include_directories(comm_scope PRIVATE 
    ${SRC_DIR}
    ${THIRDPARTY_DIR}
    ${TOP_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    "src"
)

target_compile_features(comm_scope PUBLIC cxx_std_11)
set_property(TARGET comm_scope PROPERTY CUDA_STANDARD 11)
target_link_libraries(comm_scope PRIVATE benchmark::benchmark)

## NUMA
find_package(numa)
if(numa_FOUND)
    target_compile_definitions(comm_scope PRIVATE -DUSE_NUMA=1)
    target_link_libraries(comm_scope PUBLIC numa)
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

target_compile_definitions(comm_scope PRIVATE -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
target_compile_definitions(comm_scope PRIVATE -DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})

MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
