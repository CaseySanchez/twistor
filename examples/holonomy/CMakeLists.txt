cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

project(holonomy LANGUAGES CXX CUDA)

find_package(HDF5 CONFIG REQUIRED COMPONENTS C CXX)

add_executable(holonomy "source/main.cu")

target_include_directories(holonomy PRIVATE ${HDF5_INCLUDE_DIR})
target_link_libraries(holonomy PRIVATE hdf5_cpp-static twistor)