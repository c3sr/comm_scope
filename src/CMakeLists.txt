set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES} 
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/duplex_gpu_to_gpu_peer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/duplex_gpu_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/gpu_to_gpu_peer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/gpu_to_gpu.cpp
PARENT_SCOPE)