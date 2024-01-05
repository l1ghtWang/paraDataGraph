cmake_minimum_required(VERSION 3.15)

project(paraDataGraph LANGUAGES CXX CUDA)
find_package(CUDA 11 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 86)


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lpthread;-gencode;arch=compute_86,code=sm_86;-g;-G;-lineinfo;-Xcompiler;-ggdb;-std=c++17 -O0)
else ()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lpthread;-gencode;arch=compute_86,code=sm_86;-O3;-Xcompiler;-DNDEBUG;-std=c++17 -O3)
endif ()

# GFlags
add_subdirectory(external/gflags)

set(EXTRA_LIBS gflags pthread)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(tests)

# add_executable(testGraphLoad tests/testGraphLoad.cpp)
# target_link_libraries(testGraphLoad ${EXTRA_LIBS})