cmake_minimum_required(VERSION 3.8)

if(MSVC)
    add_compile_options(/W3 /WX)
else()
    add_compile_options(-W -Wall -Werror)
endif()


sugar_include("src")

add_library(comm_scope ${COMM_SCOPE_SOURCES} ${COMM_SCOPE_CUDA_SOURCES} )

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

find_package(numa)

if(numa_FOUND)
    target_compile_definitions(comm_scope PRIVATE -DUSE_NUMA=1)
    target_link_libraries(comm_scope PUBLIC numa)
else()
target_compile_definitions(comm_scope PRIVATE -DUSE_NUMA=0)
  message(WARNING "didn't find NUMA, some benchmarks will be unavailable.")
endif(numa_FOUND)

