cmake_minimum_required(VERSION 3.10) 
project(optimization_cpp CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
    osqp-cpp
    GIT_REPOSITORY https://github.com/google/osqp-cpp.git
    GIT_TAG 43433736334d6b515ea4b0247156fea9e56c0d3f
)
FetchContent_MakeAvailable(osqp-cpp)

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
    ${osqp-cpp_SOURCE_DIR}/include
)
target_link_libraries(optimization_cpp Eigen3::Eigen osqp-cpp)


add_executable(main main.cpp)
target_link_libraries(main optimization_cpp)