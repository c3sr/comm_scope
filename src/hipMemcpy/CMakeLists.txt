if(SCOPE_USE_HIP)
    set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cpp
    PARENT_SCOPE)
endif()
