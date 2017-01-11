#ifndef SUPERVOXEL_REGISTRATION_H_
#define SUPERVOXEL_REGISTRATION_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/point_picking_event.h>

#include <fstream>
#include <boost/unordered_map.hpp>
#include <boost/filesystem.hpp>

#include "supervoxel_cluster_search.h"
#include "supervoxel_mapping.hpp"
#include "supervoxel_util.hpp"

#define NORMAL_RADIUS 0.5;
#define SEARCH_SUPERVOXEL_NN 30;
#define NORM_VR 0.1
#define NORM_R 5 // 5 meters

// Should be a factor of 1.0
#define NORM_DX 0.1
#define NORM_DY 0.1
#define NORM_DZ 0.1

// Min points to be present in supevoxel for MI consideration
#define MIN_POINTS_IN_SUPERVOXEL 30

#define PROBABILITY_OUTLIERS_SUPERVOXEL 0.35

namespace svr {

bool static _SVR_DEBUG_ = true;
typedef pcl::PointXYZRGBA PointT;
typedef pcl::Normal Normal;
typedef pcl::PointXYZRGBNormal PointTNormal;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointTNormal> PointCloudTNormal;
typedef pcl::PointCloud<Normal> PointCloudNormal;
typedef pcl::SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::KdTreeFLANN<pcl::PointXYZ> KdTreeXYZ;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

//typedef boost::unordered::unordered_map<uint, typename SData::Ptr> SVMap;
typedef std::map<uint, typename SData::Ptr> SVMap;
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
	showNormalPointCloud(PointCloudNormal::Ptr normals, PointCloudT::Ptr scan);

	void
	showTestSuperVoxel(
			int supervoxelLabel);

	void
	alignScans(Eigen::Affine3d& final_transform, Eigen::Affine3d& initial_transform);

	void
	setDebug(bool d) {
		_SVR_DEBUG_ = d;
	}

	void
	setApproximate(bool a) {
		appx = a;
	}

	void
	displayPointWithPossibleSupervoxelCorrespondences(PointT query, std::vector<int> pointIdxNKNSearchVector);

	void
	setDebugScanString(std::string s) {
		debugScanString = s;
	}

protected:

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>
	initializeVoxels();

	void
	createSuperVoxelMappingForScan1 ();

	void
	createSuperVoxelMappingForScan2 (PointCloudT::Ptr transformedScanB, PointCloudXYZ::Ptr transformedNormalsB, int iteration);

	void
	calculateSupervoxelScanBData();

	void
	createKDTreeForSupervoxels();

	void
	prepareForRegistration();

	void
	calculateScan2Normals();

private:

//	void pp_callback(const pcl::visualization::PointPickingEvent& event, void* viewer);

	void printSupervoxelMap(int iteration, std::string scan_string, Eigen::Affine3d& trans, float cost) {

		std::stringstream ss;
		// clear string
		ss.str(std::string());
		ss << "../data/DEBUG_FILES/" << scan_string;

		std::string dir = ss.str();
		if(!(boost::filesystem::exists(dir))){
			std::cout<<"Debug Directory doesn't Exist"<<std::endl;

			if (boost::filesystem::create_directory(dir))
				std::cout << "....Successfully Created !" << std::endl;
		}

		ss.str(std::string());
		ss << dir << "/ITERATION_" << iteration << "_mapping";

		std::string filename(ss.str());
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

		ss.str(std::string());
		ss << dir << "/ITERATION_" << iteration << "_normals";

		filename = ss.str();
		file.open(filename.c_str());

		PointCloudXYZ::iterator pItr;

		int counter = 0;
		for (pItr = normalsB->begin(); pItr != normalsB->end(); pItr++, counter++) {
			pcl::PointXYZ p = *pItr;
			file << "Point: " << counter+1 << std::endl;
			file << "X: " << p.x << std::endl;
			file << "Y: " << p.y << std::endl;
			file << "Z: " << p.z << std::endl;
		}

		file.close();

		ss.str(std::string());
		ss << dir << "/trans_iteration_" << iteration;

		filename = ss.str();
		file.open(filename.c_str());

		file << trans.inverse().matrix();
		file << std::endl << cost;
		file.close();

	}

	bool debug, appx;
	SupervoxelClusteringT supervoxelClustering;
	SVMap supervoxelMap;
	LabeledLeafMapT leafMapping;
	float vr, sr;
	PointCloudT::Ptr A, B;
//	PointCloudTNormal::Ptr B;
	PointCloudXYZ::Ptr normalsB;
	KdTreeXYZ supervoxelKdTree;
	std::vector<int> kdTreeLabels;
	OctreeBounds octree_bounds_;
	std::string debugScanString;
};


}

#endif
