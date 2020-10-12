set(comm_SOURCES ${comm_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/cudaMemcpy3DPeerAsync.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cudaMallocManaged.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cudaMemcpy3DAsync_gpu_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cudaMemcpyAsync_pinned_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cudaMemcpyAsync_gpu_to_pinned.cpp
PARENT_SCOPE)

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/cudaGraphInstantiate_cudaMemcpyAsync.cu
  ${CMAKE_CURRENT_LIST_DIR}/cudaGraphInstantiate_kernel.cu
  ${CMAKE_CURRENT_LIST_DIR}/cudaGraphLaunch_cudaMemcpy3DAsync.cu
  ${CMAKE_CURRENT_LIST_DIR}/cudaGraphLaunch_cudaMemcpyAsync.cu
  ${CMAKE_CURRENT_LIST_DIR}/cudaGraphLaunch_kernel.cu
  ${CMAKE_CURRENT_LIST_DIR}/kernel.cu
PARENT_SCOPE)
