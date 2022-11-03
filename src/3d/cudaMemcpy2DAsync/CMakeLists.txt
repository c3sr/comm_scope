if (SCOPE_USE_CUDA)
    set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_pinned.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pinned_to_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cpp
    PARENT_SCOPE)
endif()

set(comm_CUDA_SOURCES ${comm_CUDA_SOURCES}
PARENT_SCOPE)