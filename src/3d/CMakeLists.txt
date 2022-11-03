add_subdirectory(cudaMemcpy2DAsync)
add_subdirectory(cudaMemcpy3DAsync)
add_subdirectory(cudaMemcpy3DPeerAsync)
add_subdirectory(kernel3D)

if(SCOPE_USE_CUDA)
    set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/pack_cudaMemcpyPeerAsync_unpack.cu
    PARENT_SCOPE)
endif()

set(comm_SOURCES ${comm_SOURCES}
PARENT_SCOPE)
