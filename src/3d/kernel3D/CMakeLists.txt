set(comm_SOURCES ${comm_SOURCES}
PARENT_SCOPE)

if (SCOPE_USE_CUDA)
    set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_pinned.cu
    ${CMAKE_CURRENT_LIST_DIR}/pinned_to_gpu.cu
    ${CMAKE_CURRENT_LIST_DIR}/pull.cu
    ${CMAKE_CURRENT_LIST_DIR}/push.cu
    PARENT_SCOPE)
endif()
