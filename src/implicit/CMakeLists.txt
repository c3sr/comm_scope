
if(SCOPE_USE_HIP)
    set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/managed_gpu_read_host.cpp
    ${CMAKE_CURRENT_LIST_DIR}/managed_gpu_write_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/managed_gpu_write_host.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mapped_gpu_read_host.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mapped_gpu_write_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mapped_gpu_write_host.cpp
    PARENT_SCOPE)
endif()

if(SCOPE_USE_CUDA)
    set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
    PARENT_SCOPE)
endif()