if(SCOPE_USE_HIP)
  set(comm_SOURCES ${comm_SOURCES}
  # ${CMAKE_CURRENT_LIST_DIR}/gpu_gpu_peer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pageable.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pinned.cpp
  PARENT_SCOPE)
endif()
