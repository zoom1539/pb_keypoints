cmake_minimum_required(VERSION 3.0.0)

# CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# OpenCV
find_package(OpenCV 3.4.15 REQUIRED PATHS /root/PLATFORM/3rdparty/opencv3415)
# find_package(OpenCV REQUIRED)

# # xtensor
# find_package(xtl 0.7.2 REQUIRED PATHS /root/PLATFORM/packages/xtl/build/install)
# find_package(xtensor REQUIRED PATHS /root/PLATFORM/3rdparty/xtensor)
# message(${xtensor_INCLUDE_DIRS})
# # json
# find_package(nlohmann_json REQUIRED PATHS /root/PLATFORM/3rdparty/nlohmann_json)


# #libtorch
# set(LIBTORCH_ROOT /root/PLATFORM/3rdparty/libtorch131)
# set(LIBTORCH_INCLUDE_DIRS ${LIBTORCH_ROOT}/include ${LIBTORCH_ROOT}/include/torch/csrc/api/include)
# set(LIBTORCH_LIB_DIR ${LIBTORCH_ROOT}/lib)

# TensorRT
set(TENSORRT_ROOT /root/PLATFORM/3rdparty/TensorRT-7.0.0.11)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)

#
# aux_source_directory(. LIB_SRC)
file(GLOB_RECURSE LIB_SRC ./*.cpp ./*.cu)

# message(${LIB_SRC})
add_library(keypoints SHARED ${LIB_SRC} )

target_include_directories(keypoints 
            PUBLIC ${TENSORRT_INCLUDE_DIR}
            PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(keypoints ${OpenCV_LIBS} 
                      ${TENSORRT_LIB_DIR}/libnvinfer.so 
                      ${CUDA_LIBRARIES})

# cd build 
# make install
install(FILES class_keypoints.h DESTINATION include)
install(TARGETS keypoints LIBRARY DESTINATION lib)
install(FILES simple_res50_256x192.wts DESTINATION lib)
