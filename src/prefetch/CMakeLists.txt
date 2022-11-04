if(SCOPE_USE_HIP)
    set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/host_to_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_host.cpp
    PARENT_SCOPE)
endif()

if(SCOPE_USE_CUDA)
    set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cu
    ${CMAKE_CURRENT_LIST_DIR}/host_to_gpu.cu
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_host.cu
    PARENT_SCOPE)
endif()