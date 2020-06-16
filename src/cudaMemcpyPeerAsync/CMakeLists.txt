set(comm_SOURCES ${comm_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu_peer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cpp
PARENT_SCOPE)

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
PARENT_SCOPE)