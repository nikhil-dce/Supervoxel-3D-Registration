#ifndef SUPERVOXEL_REGISTRATION_H_
#define SUPERVOXEL_REGISTRATION_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "supervoxel_cluster_search.h"
#include "supervoxel_mapping.hpp"

#define SEARCH_SUPERVOXEL_NN 10;
#define NORM_VR 0.1
#define NORM_R 5 // 5 meters

// Should be a factor of 1.0
#define NORM_DX 0.1
#define NORM_DY 0.1
#define NORM_DZ 0.1

// Min points to be present in supevoxel for MI consideration
#define MIN_POINTS_IN_SUPERVOXEL 5

namespace svr {

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::KdTreeFLANN<pcl::PointXYZ> KdTreeXYZ;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

typedef std::map<uint, typename SData::Ptr> SVMap;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, uint32_t> LabeledLeafMapT;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> LeafVoxelMapT;
typedef typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr AdjacencyOctreeT;

struct octBounds {
	PointT minPt, maxPt;
}octree_bounds_;

struct MI_Opti_Data{
	SVMap* svMap;
	PointCloudT::Ptr scan1;
	PointCloudT::Ptr scan2;
};

class SupervoxelRegistration {

public:

	SupervoxelRegistration(float vr, float sr);

	~SupervoxelRegistration();

	void setScans(
			PointCloudT::Ptr A,
			PointCloudT::Ptr B);

	void
	showPointCloud(
			typename PointCloudT::Ptr);

	void
	showPointClouds(
			std::string viewerTitle);

	void
	showTestSuperVoxel(
			int supervoxelLabel);

	void
	alignScans();

protected:

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>
	initializeVoxels();

	void
	createSuperVoxelMappingForScan1 ();

	void
	createSuperVoxelMappingForScan2 ();

	void
	calculateSupervoxelScanBData();

	Eigen::Affine3d
	optimize();

	void
	createKDTreeForSupervoxels();

	void
	prepareForRegistration();

private:

	SupervoxelClusteringT supervoxelClustering;
	SVMap supervoxelMap;
	LabeledLeafMapT leafMapping;
	float vr, sr;
	PointCloudT::Ptr A, B;
	KdTreeXYZ supervoxelKdTree;
	std::vector<int> kdTreeLabels;
};


}

#endif
