set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu_write_dst.cpp
PARENT_SCOPE)

if(SCOPE_USE_CUDA)
    set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
    PARENT_SCOPE)
endif()