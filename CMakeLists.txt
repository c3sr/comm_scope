cmake_minimum_required(VERSION 3.8)

if(MSVC)
    add_compile_options(/W3 /WX)
else()
    add_compile_options(-W -Wall -Werror)
endif()

find_package(numa REQUIRED)

add_library(comm_scope)