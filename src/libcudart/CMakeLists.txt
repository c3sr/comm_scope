set(comm_SOURCES ${comm_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/cudaMemcpy3DPeerAsync.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cudaMallocManaged.cpp
PARENT_SCOPE)

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
${CMAKE_CURRENT_LIST_DIR}/kernel.cu
PARENT_SCOPE)