cmake_minimum_required(VERSION 3.27)
project(yolodetect)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(torch.cmake)
include(opencv.cmake)

add_subdirectory(src)