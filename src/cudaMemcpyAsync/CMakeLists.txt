set(comm_SOURCES ${comm_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu_nopeer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu_peer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_host.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_pinned.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gpu_to_wc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/host_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/host_to_pinned.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pinned_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/wc_to_gpu.cpp
PARENT_SCOPE)

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
PARENT_SCOPE)