#ifndef SUPERVOXEL_REGISTRATION_H_
#define SUPERVOXEL_REGISTRATION_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <fstream>
#include <boost/unordered_map.hpp>

#include "supervoxel_cluster_search.h"
#include "supervoxel_mapping.hpp"
#include "supervoxel_util.hpp"

#define SEARCH_SUPERVOXEL_NN 20;
#define NORM_VR 0.1
#define NORM_R 5 // 5 meters

// Should be a factor of 1.0
#define NORM_DX 0.1
#define NORM_DY 0.1
#define NORM_DZ 0.1

// Min points to be present in supevoxel for MI consideration
#define MIN_POINTS_IN_SUPERVOXEL 20

#define PROBABILITY_OUTLIERS_SUPERVOXEL 0.35

namespace svr {

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::KdTreeFLANN<pcl::PointXYZ> KdTreeXYZ;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

typedef boost::unordered::unordered_map<uint, typename SData::Ptr> SVMap;
//typedef std::map<uint, typename SData::Ptr> SVMap;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, uint32_t> LabeledLeafMapT;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> LeafVoxelMapT;
typedef typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr AdjacencyOctreeT;

struct OctreeBounds {

	PointT minPt, maxPt;

	OctreeBounds() {

	}
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

	Eigen::Matrix4d
	alignScans();

	void
	setDebug(bool d) {
		debug = d;
	}

	void
	setApproximate(bool a) {
		appx = a;
	}

protected:

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>
	initializeVoxels();

	void
	createSuperVoxelMappingForScan1 ();

	void
	createSuperVoxelMappingForScan2 (PointCloudT::Ptr transformedScan2);

	void
	calculateSupervoxelScanBData();

	void
	createKDTreeForSupervoxels();

	void
	prepareForRegistration();

private:

	void printSupervoxelMap() {

		std::string filename("Supervoxel Map ");
		std::ofstream file(filename.c_str());

		SVMap::iterator svItr;
		for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

			SData::Ptr supervoxel = svItr->second;
			file << "Label: " <<  supervoxel->getLabel() << std::endl;
			file << "A: " << supervoxel->getPointACount() << std::endl;
			file << "B: " << supervoxel->getPointBCount() << std::endl;
			file << "Covariance: " << std::endl; file << supervoxel->getCovariance() << std::endl;
			file << "Epsilons: " << supervoxel->getEpsilon1() << ' ' << supervoxel->getEpsilon2() << std::endl;

		}

		file.close();
	}

	bool debug, appx;
	SupervoxelClusteringT supervoxelClustering;
	SVMap supervoxelMap;
	LabeledLeafMapT leafMapping;
	float vr, sr;
	PointCloudT::Ptr A, B;
	KdTreeXYZ supervoxelKdTree;
	std::vector<int> kdTreeLabels;
	OctreeBounds octree_bounds_;
};


}

#endif
