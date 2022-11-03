add_subdirectory(3d)

message(STATUS "comm_SOURCES ${comm_SOURCES}")
message(STATUS "comm_CUDA_SOURCES ${comm_CUDA_SOURCES}")


add_subdirectory(chunk)
add_subdirectory(cudaMemcpyAsync)
add_subdirectory(cudaMemcpyAsync-duplex)
add_subdirectory(cudaMemcpyPeerAsync)
add_subdirectory(cudaMemcpyPeerAsync-duplex)
add_subdirectory(demand)
add_subdirectory(demand-duplex)

message(STATUS "comm_SOURCES ${comm_SOURCES}")
message(STATUS "comm_CUDA_SOURCES ${comm_CUDA_SOURCES}")

add_subdirectory(hipMemcpy)

message(STATUS "comm_SOURCES ${comm_SOURCES}")
message(STATUS "comm_CUDA_SOURCES ${comm_CUDA_SOURCES}")

add_subdirectory(latency)
add_subdirectory(libc)
add_subdirectory(libcudart)
add_subdirectory(prefetch)
add_subdirectory(prefetch-duplex)
add_subdirectory(stride)
add_subdirectory(zerocopy)
add_subdirectory(zerocopy-duplex)

set(comm_SOURCES ${comm_SOURCES} PARENT_SCOPE)
set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES} PARENT_SCOPE)
