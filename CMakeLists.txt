cmake_minimum_required(VERSION 3.0.0)
project(opencv-gpu-test VERSION 0.1.0)

include(CTest)
enable_testing()

set (OpenCV_DIR C:/Users/zachf/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( &{OpenCV_INCLUDE_DIRS} )
add_executable( RedDetection reddetection.cpp )
target_link_libraries( RedDetection ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

