if (SCOPE_USE_HIP)
  set(comm_SOURCES ${comm_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/hipDeviceSynchronize.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel.cpp
  PARENT_SCOPE)
endif()