cmake_minimum_required(VERSION 3.9)

# SET(CMAKE_BUILD_TYPE RelWithDebInfo)
SET(CMAKE_BUILD_TYPE Release)

project(alignment)

find_package(PCL 1.7 REQUIRED)
find_package(Eigen REQUIRED)
find_package(pybind11)

#cuda info:
# http://stackoverflow.com/questions/13073717/building-cuda-object-files-using-cmake
# https://devtalk.nvidia.com/default/topic/487573/cuda-programming-and-performance/linker-error-usr-bin-ld-cannot-find-lcudart/post/3497498/#3497498
find_package(CUDA 9.0 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)
set(CUDA_VERBOSE_BUILD ON)

FILE(GLOB srcs src/*.cpp)
FILE(GLOB cuda src/cuda/*.cu)

source_group("Source Files\\cuda" FILES ${cuda})
source_group("Source Files" FILES ${srcs})

set(LIB_NAME alignment_cuda)

find_library(CUDADEVRT
   NAMES cudadevrt
   PATHS "/opt/" "/usr/local/" 
   PATH_SUFFIXES "cuda/lib64"
   DOC "Location of cudadevrt library"
   NO_DEFAULT_PATH
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ -D_FORCE_INLINES -O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -O0 -g --std=c++11")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ftz=true;--prec-div=false;--prec-sqrt=false;-arch=sm_35;--expt-extended-lambda)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DTHRUST_DEBUG;-g)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_64_BIT_DEVICE_CODE ON)

CUDA_ADD_LIBRARY(alignment_lib ${cuda} ${srcs})
target_include_directories(alignment_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PCL_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
    )

add_library (alignment SHARED src/alignment.cpp)
target_include_directories(alignment PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PCL_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
    )
target_link_libraries (alignment alignment_lib ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

pybind11_add_module(example alignment_py.cpp)
