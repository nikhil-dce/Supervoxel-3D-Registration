cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(supervoxel_EM)

find_package(PCL 1.7 REQUIRED COMPONENTS common io visualization octree)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(GSL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (registration main.cpp supervoxel_registration.cpp supervoxel_cluster_search.cpp supervoxel_mapping.cpp supervoxel_octree_pointcloud_adjacency.hpp supervoxel_util.hpp supervoxel_optimize.h)
 
target_link_libraries (registration ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES})
