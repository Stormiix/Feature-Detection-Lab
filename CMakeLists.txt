cmake_minimum_required(VERSION 2.8.3)
SET("OpenCV_DIR" "C:/opencv/build/x64/Release/lib")
SET("OpenCV_INCLUDE" "C:/opencv/build/include")
INCLUDE_DIRECTORIES("include" "${OpenCV_INCLUDE}/")
project( tracking )
find_package( OpenCV REQUIRED )
add_executable( tracking main/tracking.cpp )
add_executable( trackingF main/trackingF.cpp )
add_executable( trackingF_KNN main/trackingF_KNN.cpp )
target_link_libraries( tracking ${OpenCV_DIR} )
target_link_libraries( trackingF ${OpenCV_DIR} )
target_link_libraries( trackingF_KNN ${OpenCV_DIR} )


