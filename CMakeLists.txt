cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

project(twistor LANGUAGES CXX CUDA)

find_package(GATL REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(twistor INTERFACE)

target_compile_options(twistor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(twistor INTERFACE "include" ${GATL_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(twistor INTERFACE cuda)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()