if(SCOPE_USE_CUDA)
    set(comm_SOURCES ${comm_SOURCES}
    PARENT_SCOPE)

    set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_gpu.cu
    ${CMAKE_CURRENT_LIST_DIR}/host_gpu.cu
    PARENT_SCOPE)
endif()