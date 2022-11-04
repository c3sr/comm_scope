if(SCOPE_USE_HIP)
    set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/numa_to_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cpp
    PARENT_SCOPE)
endif()
