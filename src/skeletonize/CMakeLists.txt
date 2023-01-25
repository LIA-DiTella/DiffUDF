cmake_minimum_required (VERSION 3.22.1)
project (Skeleton)

set(CMAKE_BUILD_TYPE RELEASE)
find_package(CGAL REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(Skel skeletonize.cpp)

target_link_libraries(Skel CGAL::CGAL)
target_link_libraries(Skel Eigen3::Eigen)
