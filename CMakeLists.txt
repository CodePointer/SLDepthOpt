cmake_minimum_required(VERSION 3.9)
project(DepthOptimization)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
        optimizer
        global
        ${OPENCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Ceres_INCLUDE_DIRS}
)

add_executable(DepthOptimization
        main.cpp
        optimizer/image_cost_functor.h
        optimizer/image_cost_functor.cpp
        global/static_para.h
        optimizer/image_optimizer.cpp
        optimizer/image_optimizer.h optimizer/regular_cost_functor.cpp optimizer/regular_cost_functor.h)

target_link_libraries(DepthOptimization
        ceres
        ${OPENCV_LIBS})