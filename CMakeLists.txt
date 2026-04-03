cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

option(BUILD_HDF5 "Build HDF5 read/write support" OFF)
option(BUILD_EXAMPLES "Build the examples" OFF)

project(twistor LANGUAGES CXX CUDA)

find_package(GATL REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(BUILD_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
endif()

add_library(twistor INTERFACE)

target_compile_options(twistor INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(twistor INTERFACE "include/" ${GATL_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(twistor INTERFACE cuda)

if(BUILD_HDF5)
    target_compile_definitions(twistor INTERFACE TWISTOR_HDF5)
    target_include_directories(twistor INTERFACE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(twistor INTERFACE ${HDF5_CXX_LIBRARIES})
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()