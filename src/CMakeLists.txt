set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES} 
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/duplex_gpu_to_gpu_peer.cu
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/duplex_gpu_to_gpu.cu
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/gpu_to_gpu_peer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/memcpypeer/gpu_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy3DPeerAsync.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy2DAsync.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/push.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/pull.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/numa_to_gpu.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/kernel3D/gpu_to_numa.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/pack_cudaMemcpyPeerAsync_unpack.cu
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy2DAsync/gpu_to_numa.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy2DAsync/numa_to_gpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy3DAsync/gpu_to_numa.cpp
  ${CMAKE_CURRENT_LIST_DIR}/3d/cudaMemcpy3DAsync/numa_to_gpu.cpp
PARENT_SCOPE)
