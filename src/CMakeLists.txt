add_subdirectory(cudaMemcpyAsync)
add_subdirectory(cudaMemcpyAsync-duplex)
add_subdirectory(cudaMemcpyPeerAsync)
add_subdirectory(cudaMemcpyPeerAsync-duplex)
add_subdirectory(demand)
add_subdirectory(demand-duplex)
add_subdirectory(latency)
add_subdirectory(libc)
add_subdirectory(libcudart)
add_subdirectory(prefetch)
add_subdirectory(prefetch-duplex)
add_subdirectory(zerocopy)
add_subdirectory(zerocopy-duplex)

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/push.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/pull.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/numa_to_gpu.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/gpu_to_numa.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/pack_cudaMemcpyPeerAsync_unpack.cu
PARENT_SCOPE)

set(comm_SOURCES ${comm_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy2DAsync/gpu_to_numa.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy2DAsync/gpu_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy2DAsync/numa_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy3DAsync/gpu_to_numa.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy3DAsync/numa_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy3DPeerAsync/gpu_to_gpu.cpp
PARENT_SCOPE)


