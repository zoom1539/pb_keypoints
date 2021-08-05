cmake_minimum_required(VERSION 3.0.0)
project(demo VERSION 0.1.0)

#
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_PREFIX install)

# OpenCV
find_package(OpenCV 3.4.15 REQUIRED PATHS /root/PLATFORM/3rdparty/opencv3415)
# find_package(OpenCV REQUIRED)


#
add_subdirectory(lib)
aux_source_directory(. SRC)
add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} 
            PUBLIC ${OpenCV_INCLUDE_DIRS}
            PUBLIC lib)

target_link_libraries(${PROJECT_NAME} keypoints)

#install
install(FILES main.cpp DESTINATION .)



