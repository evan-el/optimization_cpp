cmake_minimum_required(VERSION 3.10) 
project(optimization_cpp CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(OPTIMIZATION_CPP_SRCS "")
file(GLOB_RECURSE OPTIMIZATION_CPP_SRCS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

add_library(optimization_cpp ${OPTIMIZATION_CPP_SRCS})
target_include_directories(optimization_cpp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(optimization_cpp Eigen3::Eigen)


add_executable(main main.cpp)
target_link_libraries(main optimization_cpp)