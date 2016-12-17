cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(supervoxel_mi)

find_package(PCL 1.7 REQUIRED COMPONENTS common io visualization octree)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(GSL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add_executable (supervoxel_mi supervoxel_mi.cpp supervoxel_cluster_search.cpp supervoxel_mapping.cpp supervoxel_octree_pointcloud_adjacency.hpp)
add_executable (supervoxel_mi_smooth supervoxel_mi_smooth.cpp supervoxel_cluster_search.cpp supervoxel_mapping.cpp supervoxel_octree_pointcloud_adjacency.hpp)
 
# target_link_libraries (supervoxel_mi ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES})
target_link_libraries (supervoxel_mi_smooth ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIBRARIES})
