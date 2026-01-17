cmake_minimum_required(VERSION 3.18)
project(posebyte_tracker CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT paths (adjust as needed)
set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation path")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT to the installation path.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/kalman_filter.cu
    src/cuda/oks_distance.cu
    src/cuda/hungarian.cu
    src/cuda/nms.cu
    src/cuda/preprocess.cu
    src/cuda/gpu_postprocess.cu
    src/cuda/gpu_tracker.cu
)

# C++ source files
set(CPP_SOURCES
    src/tensorrt/yolo_pose_engine.cpp
    src/utils/video_utils.cpp
)

# Create library
add_library(posebyte_lib STATIC
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

target_link_libraries(posebyte_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_LIBRARY}
    cudart
    cublas
)

set_target_properties(posebyte_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89"  # Adjust for your GPU
)

# Main executable
add_executable(posebyte_demo src/main.cpp)
target_link_libraries(posebyte_demo posebyte_lib)

# Benchmark executable
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark posebyte_lib)

# Export engine utility
add_executable(export_engine src/export_engine.cpp)
target_link_libraries(export_engine posebyte_lib)

# Install
install(TARGETS posebyte_demo posebyte_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
