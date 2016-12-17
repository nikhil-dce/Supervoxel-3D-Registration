#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>

#include "supervoxel_octree_pointcloud_adjacency.h"
#include "supervoxel_cluster_search.hpp"


namespace pcl
{
  namespace octree
  {
    //Explicit overloads for RGB types
    template<>
    void
    pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGB,pcl::SupervoxelClustering<pcl::PointXYZRGB>::VoxelData>::addPoint (const pcl::PointXYZRGB &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
      //Separate sums for r,g,b since we cant sum in uchars
      data_.rgb_[0] += static_cast<float> (new_point.r);
      data_.rgb_[1] += static_cast<float> (new_point.g);
      data_.rgb_[2] += static_cast<float> (new_point.b);
    }

    template<>
    void
    pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGBA,pcl::SupervoxelClustering<pcl::PointXYZRGBA>::VoxelData>::addPoint (const pcl::PointXYZRGBA &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
      //Separate sums for r,g,b since we cant sum in uchars
      data_.rgb_[0] += static_cast<float> (new_point.r);
      data_.rgb_[1] += static_cast<float> (new_point.g);
      data_.rgb_[2] += static_cast<float> (new_point.b);
    }



    //Explicit overloads for RGB types
    template<> void
    pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGB,pcl::SupervoxelClustering<pcl::PointXYZRGB>::VoxelData>::computeData ()
    {
      data_.rgb_[0] /= (static_cast<float> (num_points_));
      data_.rgb_[1] /= (static_cast<float> (num_points_));
      data_.rgb_[2] /= (static_cast<float> (num_points_));
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

    template<> void
    pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGBA,pcl::SupervoxelClustering<pcl::PointXYZRGBA>::VoxelData>::computeData ()
    {
      data_.rgb_[0] /= (static_cast<float> (num_points_));
      data_.rgb_[1] /= (static_cast<float> (num_points_));
      data_.rgb_[2] /= (static_cast<float> (num_points_));
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

    //Explicit overloads for XYZ types
    template<>
    void
    pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZ,pcl::SupervoxelClustering<pcl::PointXYZ>::VoxelData>::addPoint (const pcl::PointXYZ &new_point)
    {
      ++num_points_;
      //Same as before here
      data_.xyz_[0] += new_point.x;
      data_.xyz_[1] += new_point.y;
      data_.xyz_[2] += new_point.z;
    }

    template<> void
    pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZ,pcl::SupervoxelClustering<pcl::PointXYZ>::VoxelData>::computeData ()
    {
      data_.xyz_[0] /= (static_cast<float> (num_points_));
      data_.xyz_[1] /= (static_cast<float> (num_points_));
      data_.xyz_[2] /= (static_cast<float> (num_points_));
    }

  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace pcl
{
  template<> void
  pcl::SupervoxelClustering<pcl::PointXYZRGB>::VoxelData::getPoint (pcl::PointXYZRGB &point_arg) const
  {
    point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 |
    static_cast<uint32_t>(rgb_[1]) << 8 |
    static_cast<uint32_t>(rgb_[2]);
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
  }

  template<> void
  pcl::SupervoxelClustering<pcl::PointXYZRGBA>::VoxelData::getPoint (pcl::PointXYZRGBA &point_arg ) const
  {
    point_arg.rgba = static_cast<uint32_t>(rgb_[0]) << 16 |
    static_cast<uint32_t>(rgb_[1]) << 8 |
    static_cast<uint32_t>(rgb_[2]);
    point_arg.x = xyz_[0];
    point_arg.y = xyz_[1];
    point_arg.z = xyz_[2];
  }
}

typedef pcl::SupervoxelClustering<pcl::PointXYZ>::VoxelData VoxelDataT;
typedef pcl::SupervoxelClustering<pcl::PointXYZRGB>::VoxelData VoxelDataRGBT;
typedef pcl::SupervoxelClustering<pcl::PointXYZRGBA>::VoxelData VoxelDataRGBAT;

typedef pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZ, VoxelDataT> AdjacencyContainerT;
typedef pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGB, VoxelDataRGBT> AdjacencyContainerRGBT;
typedef pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGBA, VoxelDataRGBAT> AdjacencyContainerRGBAT;

template class pcl::SupervoxelClustering<pcl::PointXYZ>;
template class pcl::SupervoxelClustering<pcl::PointXYZRGB>;
template class pcl::SupervoxelClustering<pcl::PointXYZRGBA>;

template class pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZ, VoxelDataT>;
template class pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGB, VoxelDataRGBT>;
template class pcl::octree::OctreePointCloudAdjacencyContainer<pcl::PointXYZRGBA, VoxelDataRGBAT>;

template class pcl::octree::OctreePointCloudAdjacency<pcl::PointXYZ, AdjacencyContainerT>;
template class pcl::octree::OctreePointCloudAdjacency<pcl::PointXYZRGB, AdjacencyContainerRGBT>;
template class pcl::octree::OctreePointCloudAdjacency<pcl::PointXYZRGBA, AdjacencyContainerRGBAT>;
