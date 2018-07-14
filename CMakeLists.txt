cmake_minimum_required(VERSION 3.1)

project(tf_inference_gui)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

# Find the QtWidgets library
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)


add_executable(${PROJECT_NAME} "main.cpp" "mainwindow.h" "mainwindow.cpp")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC  ")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} " )

include_directories("/usr/local/tensorflow/include/external/nsync/public")
include_directories("/usr/local/tensorflow/include/")

TARGET_LINK_LIBRARIES(${PROJECT_NAME}  "/usr/local/tensorflow/lib/libtensorflow_cc.so")
TARGET_LINK_LIBRARIES(${PROJECT_NAME}  "/usr/local/tensorflow/lib/libtensorflow_framework.so")
# link required libs
target_link_libraries(tf_inference_gui Qt5::Core Qt5::Gui Qt5::Widgets)
