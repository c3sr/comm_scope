if(SCOPE_USE_HIP)
    set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/pageable_to_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pinned_to_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_pageable.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_pinned.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_to_gpu.cpp
    PARENT_SCOPE)
endif()
