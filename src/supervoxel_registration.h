#ifndef SUPERVOXEL_REGISTRATION_H_
#define SUPERVOXEL_REGISTRATION_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "supervoxel_cluster_search.h"

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
}octreeBounds;


class SupevoxelRegistration {

public:

	void alignScans(PointCloudT::Ptr A, PointCloudT::Ptr B);

protected:

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>
	initializeVoxels(
			PointCloudT::Ptr scan1,
			PointCloudT::Ptr scan2,
			SupervoxelClusteringT& super,
			SVMap& supervoxelMapping,
			LabeledLeafMapT& labeledLeafMap);

};


}
