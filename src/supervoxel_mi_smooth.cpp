/**
 * @file supervoxel_mi.cpp
 *
 *
 * @author Nikhil Mehta
 * @date 2016-11-21
 */

#include <string>
#include <boost/thread/thread.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/segmentation/supervoxel_clustering.h>

#include "supervoxel_mapping.hpp"
#include <cmath>
#include <iomanip>


#include <gsl/gsl_multimin.h>
#include "supervoxel_octree_pointcloud_adjacency.h"
#include <cmath>

using namespace pcl;
using namespace std;

typedef PointXYZRGBA PointT;
typedef PointCloud<PointT> PointCloudT;
typedef SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef PointXYZL PointLT;
typedef PointCloud<PointLT> PointLCloudT;
typedef KdTreeFLANN<PointXYZ> KdTreeXYZ;
typedef PointCloud<PointXYZ> PointCloudXYZ;
//typedef typename std::vector<typename SuperVoxelMapping::Ptr> SVMappingVector;

// Replace map with boost::unordered_map for faster lookup
typedef std::map<uint, typename SData::Ptr> SVMap;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, uint32_t> LabeledLeafMapT;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> LeafVoxelMapT;
typedef typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr AdjacencyOctreeT;

#define SEARCH_SUPERVOXEL_NN 10;
#define NORM_VR 0.1
#define NORM_R 5 // 5 meters

// Should be a factor of 1.0
#define NORM_DX 0.1
#define NORM_DY 0.1
#define NORM_DZ 0.1

// Min points to be present in supevoxel for MI consideration
#define MIN_POINTS_IN_SUPERVOXEL 5

struct options {

	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;
	bool showScans;

}programOptions;

struct octBounds {
	PointT minPt, maxPt;
}octreeBounds;

template <typename Type>
inline Type square(Type x)
{
	return x * x; // will only work on types for which there is the * operator.
}

void genOctreeKeyforPoint(const typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr adjTree, const PointT& point_arg, SuperVoxelMappingHelper::OctreeKeyT & key_arg) {
	// calculate integer key for point coordinates

	double min_x, min_y, min_z;
	double max_x, max_y, max_z;

	adjTree -> getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
	double resolution = adjTree->getResolution();

	key_arg.x = static_cast<unsigned int> ((point_arg.x - min_x) / resolution);
	key_arg.y = static_cast<unsigned int> ((point_arg.y - min_y) / resolution);
	key_arg.z = static_cast<unsigned int> ((point_arg.z - min_z) / resolution);
}

map <uint32_t, Supervoxel<PointT>::Ptr>
initializeVoxels(
		PointCloudT::Ptr scan1,
		PointCloudT::Ptr scan2,
		SupervoxelClusteringT& super,
		SVMap& supervoxelMapping,
		LabeledLeafMapT& labeledLeafMap);

void
showPointCloud(
		typename PointCloudT::Ptr);

void
showPointClouds(
		PointCloudT::Ptr,
		PointCloudT::Ptr,
		string viewerTitle);

void
showTestSuperVoxel(
		map<uint, typename SData::Ptr>& SVMapping,
		PointCloudT::Ptr scan1,
		PointCloudT::Ptr scan2);

double
calculateMutualInformation(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

void
createSuperVoxelMappingForScan1 (
		SVMap& SVMapping,
		const typename PointCloudT::Ptr scan,
		LabeledLeafMapT& labeledLeafMapping,
		const AdjacencyOctreeT adjTree);

void
createSuperVoxelMappingForScan2 (
		SVMap& SVMapping,
		const typename PointCloudT::Ptr scan,
		LabeledLeafMapT& labeledLeafMapping,
		//		const AdjacencyOctreeT adjTree2,
		const AdjacencyOctreeT adjTree1,
		KdTreeXYZ& svKdTree,
		std::vector<int>& treeLables);

void
calculateSupervoxelScanBData(
		SVMap& supervoxelMapping,
		PointCloudT::Ptr scanB);

Eigen::Affine3d
optimize(
		SVMap& SVMapping,
		PointCloudT::Ptr scan1,
		PointCloudT::Ptr scan2,
		gsl_vector* basePose);

Eigen::Vector4f
getNormalizedVectorCode(Eigen::Vector3f vector);

int
getCentroidResultantCode(double norm);

int
getVarianceCode(double norm);

void
createKDTreeForSupervoxels(
		SVMap& supervoxelMap,
		KdTreeXYZ& svKdTree,
		std::vector<int>& labels);

void
transform_get_translation(Eigen::Matrix4d t, double *x, double *y, double *z) {

	*x = t(0,3);
	*y = t(1,3);
	*z = t(2,3);

}

void
transform_get_rotation(Eigen::Matrix4d t, double *x, double *y, double *z) {

	double a = t(2,1);
	double b = t(2,2);
	double c = t(2,0);
	double d = t(1,0);
	double e = t(0,0);

	*x = atan2(a, b);
	*y = asin(-c);
	*z = atan2(d, e);

}

void
printPointClouds(PointCloudT::Ptr scanA, PointCloudT::Ptr transformedScan, string filename) {

	ofstream fout(filename.c_str());

	if (scanA->size() != transformedScan->size()) {
		cerr << "Scan Length not same " << endl;
		return;
	}

	for (int i = 0; i < scanA->size(); ++i) {
		fout << scanA->at(i).x << ' ' << scanA->at(i).y << ' ' << scanA->at(i).z << '\t' << transformedScan->at(i).x << ' ' << transformedScan->at(i).y << ' ' << transformedScan->at(i).z;
		fout << endl;
	}

	fout.close();
}


int initOptions(int argc, char* argv[]) {

	namespace po = boost::program_options;

	programOptions.colorWeight = 0.0f;
	programOptions.spatialWeight = 0.4f;
	programOptions.normalWeight = 1.0f;
	programOptions.vr = 1.0f;
	programOptions.sr = 5.0f;
	programOptions.test = 0; // 607 476
	programOptions.showScans = false;

	po::options_description desc ("Allowed Options");

	desc.add_options()
								("help,h", "Usage <Scan 1 Path> <Scan 2 Path> <Transform File>")
								("voxel_res,v", po::value<float>(&programOptions.vr), "voxel resolution")
								("seed_res,s", po::value<float>(&programOptions.sr), "seed resolution")
								("color_weight,c", po::value<float>(&programOptions.colorWeight), "color weight")
								("spatial_weight,z", po::value<float>(&programOptions.spatialWeight), "spatial weight")
								("normal_weight,n", po::value<float>(&programOptions.normalWeight), "normal weight")
								("test,t", po::value<int>(&programOptions.test), "test")
								("show_scan,y", po::value<bool>(&programOptions.showScans), "Show scans");

	po::variables_map vm;

	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << "Supervoxel MI based SCan Alignment" << endl << desc << endl;
		return 1;
	} else {

		cout << "vr: " << programOptions.vr << endl;
		cout << "sr: " << programOptions.sr << endl;
		cout << "colorWeight: " << programOptions.colorWeight << endl;
		cout << "spatialWeight: " << programOptions.spatialWeight << endl;
		cout << "normalWeight: " << programOptions.normalWeight << endl;

		return 0;
	}

}


int
main (int argc, char *argv[]) {

	if (initOptions(argc, argv))
		return 1;

	if (argc < 3) {
		cerr << "One or more scan files/transform missing";
		return 1;
	}

	int s1, s2;
	string transformFile;

	s1 = atoi(argv[1]);
	s2 = atoi(argv[2]);
	transformFile = argv[3];

	octreeBounds.minPt.x = -120;
	octreeBounds.minPt.y = -120;
	octreeBounds.minPt.z = -20;

	octreeBounds.maxPt.x = 120;
	octreeBounds.maxPt.y = 120;
	octreeBounds.maxPt.z = 20;


	const string dataDir = "../../Data/scans_pcd/scan_";

	PointCloudT::Ptr scan1 = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudT::Ptr scan2 = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudT::Ptr temp = boost::shared_ptr <PointCloudT> (new PointCloudT ());

	cout<<"Loading PointClouds..."<<endl;

	stringstream ss;

	ss << dataDir << boost::format("%04d.pcd")%s1;

	if (io::loadPCDFile<PointT> (ss.str(), *scan1)) {
		cout << "Error loading cloud file: " << ss.str() << endl;
		return (1);
	}

	ss.str(string());
	ss << dataDir << boost::format("%04d.pcd")%s2;

	gsl_vector *base_pose;
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();

	if (io::loadPCDFile<PointT> (ss.str(), *scan2)) {
		cout << "Error loading cloud file: " << ss.str() << endl;
		return (1);
	} else {

		ifstream in(transformFile.c_str());
		if (!in) {
			stringstream err;
			err << "Error loading transformation " << transformFile.c_str() << endl;
			cerr<<err.str();
			return 1;
		}

		string line;

		for (int i = 0; i < 4; i++) {
			std::getline(in,line);

			std::istringstream sin(line);
			for (int j = 0; j < 4; j++) {
				sin >> transform (i,j);
			}
		}

		cout << "Transformation loaded: " << endl << transform.matrix();
		cout << endl;

		if (!programOptions.showScans && programOptions.test == 0) {

			double x, y, z, roll, pitch, yaw;
			transform_get_translation(transform.matrix(), &x, &y, &z);
			transform_get_rotation(transform.matrix(), &roll, &pitch, &yaw);

			cout << "Calculating base pose" << endl;

			base_pose = gsl_vector_alloc (6);
			gsl_vector_set (base_pose, 0, x);
			gsl_vector_set (base_pose, 1, y);
			gsl_vector_set (base_pose, 2, z);
			gsl_vector_set (base_pose, 3, roll);
			gsl_vector_set (base_pose, 4, pitch);
			gsl_vector_set (base_pose, 5, yaw);

			temp = scan2;

		} else {
			// Input transform should be B rel to A
			transformPointCloud (*scan2, *temp, transform.inverse());
			scan2->clear();
		}
	}

	if (programOptions.showScans && programOptions.test == 0) {
		showPointClouds(scan1, temp, "Supervoxel Based MI Viewer: " + transformFile);
		return 0;
	}

	SupervoxelClusteringT super (programOptions.vr, programOptions.sr);
	SVMap supervoxelMapping;
	LabeledLeafMapT labeledLeafMapScan1;

	map <uint32_t, Supervoxel<PointT>::Ptr> supervoxelClusters = initializeVoxels(scan1, temp, super, supervoxelMapping, labeledLeafMapScan1);
	cout << "Number of supervoxels: " << supervoxelMapping.size() << endl;

	KdTreeXYZ svTree;
	std::vector<int> labels;
	createKDTreeForSupervoxels(supervoxelMapping, svTree, labels);

	AdjacencyOctreeT adjTreeScan1 = super.getOctreeeAdjacency();

	//	AdjacencyOctreeT adjTreeScan2;
	//	adjTreeScan2.reset (new typename SupervoxelClusteringT::OctreeAdjacencyT(programOptions.vr));
	//	adjTreeScan2->setInputCloud(temp);
	//
	//	adjTreeScan2->customBoundingBox(octreeBounds.minPt.x, octreeBounds.minPt.y, octreeBounds.minPt.z,
	//			octreeBounds.maxPt.x, octreeBounds.maxPt.y, octreeBounds.maxPt.z);
	//	adjTreeScan2->addPointsFromInputCloud();

	if (programOptions.showScans && programOptions.test != 0) {
		createSuperVoxelMappingForScan2(supervoxelMapping, temp, labeledLeafMapScan1, adjTreeScan1, svTree, labels);
		showTestSuperVoxel(supervoxelMapping, scan1, temp);
	} else if (programOptions.test != 0) {
		createSuperVoxelMappingForScan2(supervoxelMapping, temp, labeledLeafMapScan1, adjTreeScan1, svTree, labels);
		calculateSupervoxelScanBData(supervoxelMapping, temp);
		double mi = calculateMutualInformation(supervoxelMapping, scan1, temp);
		cout << "MI: " << mi << endl;
	} else {

		Eigen::Affine3d trans_last = transform;
		Eigen::Affine3d trans_new;
		PointCloudT::Ptr transformedScan2 = boost::shared_ptr <PointCloudT> (new PointCloudT ());

		bool converged = false;
		int iteration = 0;
		double delta;
		bool debug = false;
		double epsilon = 5e-4;
		double epsilon_rot = 2e-3;
		int maxIteration = 200;

		while (!converged) {

			// transform point cloud using trans_last

			transformPointCloud (*temp, *transformedScan2, trans_last);
			createSuperVoxelMappingForScan2(supervoxelMapping, transformedScan2, labeledLeafMapScan1 , adjTreeScan1, svTree, labels);

			trans_new = optimize(supervoxelMapping, scan1, temp, base_pose);

			/* compute the delta from this iteration */
			delta = 0.;
			for(int k = 0; k < 4; k++) {
				for(int l = 0; l < 4; l++) {

					double ratio = 1;
					if(k < 3 && l < 3) {
						// rotation part of the transform
						ratio = 1./epsilon_rot;
					} else {
						ratio = 1./epsilon;
					}

					double diff = trans_last.matrix()(k,l) - trans_new.matrix()(k,l);
					double c_delta = ratio*fabs(diff);

					if(c_delta > delta) {
						delta = c_delta;
					}
				}
			}

			/* check convergence */
			iteration++;
			cout << "Iteration: " << iteration << " delta = " << delta << endl;

			if(iteration >= maxIteration || delta < 1) {
				converged = true;
			}

			trans_last = trans_new;
		}

		// Save transformation in a file

		cout << "Resultant Transformation: " << endl << trans_new.inverse().matrix();
		cout << endl;

		string transFilename = (boost::format("trans_mi_%d_%d")%s1%s2).str();
		ofstream fout(transFilename.c_str());
		fout << trans_new.inverse().matrix();
		fout.close();

	}
}


map <uint32_t, Supervoxel<PointT>::Ptr>
initializeVoxels(
		PointCloudT::Ptr scan1,
		PointCloudT::Ptr scan2,
		SupervoxelClusteringT& super,
		SVMap& supervoxelMapping,
		LabeledLeafMapT& labeledLeafMap) {

	//	Eigen::Array4f min1, max1, min2, max2;
	//
	//	PointT minPt, maxPt;
	//
	//	getMinMax3D(*scan1, minPt, maxPt);
	//
	//	min1 = minPt.getArray4fMap();
	//	max1 = maxPt.getArray4fMap();
	//
	//	getMinMax3D(*scan2, minPt, maxPt);
	//
	//	min2 = minPt.getArray4fMap();
	//	max2 = maxPt.getArray4fMap();
	//
	//	min1 = min1.min(min2);
	//	max1 = max1.max(max2);
	//
	//	minPt.x = min1[0]; minPt.y = min1[1]; minPt.z = min1[2];
	//	maxPt.x = max1[0]; maxPt.y = max1[1]; maxPt.z = max1[2];
	//
	//	cout << "MinX: " << minPt.x << " MinY: " << minPt.y << " MinZ: " << minPt.z  << endl;
	//	cout << "MaxX: " << maxPt.x << " MaxY: " << maxPt.y << " MaxZ: " << maxPt.z  << endl;

	super.setVoxelResolution(programOptions.vr);
	super.setSeedResolution(programOptions.sr);
	super.setInputCloud(scan1);
	super.setColorImportance(programOptions.colorWeight);
	super.setSpatialImportance(programOptions.spatialWeight);
	super.setNormalImportance(programOptions.normalWeight);

	//	super.getOctreeeAdjacency()->customBoundingBox(minPt.x, minPt.y, minPt.z,
	//			maxPt.x, maxPt.y, maxPt.z);

	super.getOctreeeAdjacency()->customBoundingBox(octreeBounds.minPt.x, octreeBounds.minPt.y, octreeBounds.minPt.z,
			octreeBounds.maxPt.x, octreeBounds.maxPt.y, octreeBounds.maxPt.z);



	// Not being used for now
	map <uint32_t, Supervoxel<PointT>::Ptr> supervoxelClusters;
	super.extract(supervoxelClusters);
	super.getLabeledLeafContainerMap(labeledLeafMap);

	createSuperVoxelMappingForScan1(supervoxelMapping, scan1, labeledLeafMap, super.getOctreeeAdjacency());

	return supervoxelClusters;
}


void
createSuperVoxelMappingForScan1 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT adjTree) {

	PointCloud<PointT>::iterator scanItr;
	int scanCounter = 0;

	LeafVoxelMapT leafVoxelMap;

	for (scanItr = scan->begin(); scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			if (leafVoxelMap.find(leaf) != leafVoxelMap.end()) {
				leafVoxelMap[leaf]->getIndexVector()->push_back(scanCounter);
			} else {
				VData::Ptr voxel = boost::shared_ptr<VData>(new VData());
				voxel->getIndexVector()->push_back(scanCounter);
				leafVoxelMap.insert(pair<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> (leaf, voxel));
			}

		} else {
			// never be the case
			cout << "Not present in voxel"<<endl;
		}
	}

	// leafVoxelMap created for scan1

	LeafVoxelMapT::iterator leafVoxelItr;
	PointCloudT centroidCloud;
	int centroidCloudCounter = 0;

	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr, ++centroidCloudCounter) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;

		// compute Centroid
		typename VData::ScanIndexVectorPtr scanIndexVector = voxel->getIndexVector();
		typename VData::ScanIndexVector::iterator indexItr;

		PointT centroid;
		double x(0), y(0), z(0), r(0), g(0), b(0);
		for (indexItr = scanIndexVector->begin(); indexItr != scanIndexVector->end(); ++indexItr) {
			PointT p = scan->at(*indexItr);
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int size = scanIndexVector->size();;
		centroid.x = x / size;
		centroid.y = y / size;
		centroid.z = z / size;
		centroid.r = r / size;
		centroid.g = g / size;
		centroid.g = b / size;

		centroidCloud.push_back(centroid);
		voxel->setCentroid(centroid);
		voxel->setCentroidCloudIndex(centroidCloudCounter);
	}

	cout << "Finding mapping " << endl;
	cout << "Total leaves: " << leafVoxelMap.size() << endl;
	cout << "Total leaves with supervoxels " << labeledLeafMapping.size() << endl;

	int leavesNotFoundWithSupervoxel = 0;
	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;
		typename SupervoxelClusteringT::LeafContainerT::const_iterator leafItr;

		// check if leaf exists in the mapping from leaf to label
		if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

			unsigned int label = labeledLeafMapping[leaf];

			// calculate normal for this leaf
			Eigen::Vector4f params = Eigen::Vector4f::Zero();
			float curvature;
			std::vector<int> indicesToConsider;

			for (leafItr = leaf->cbegin(); leafItr != leaf->cend(); ++leafItr) {
				typename SupervoxelClusteringT::LeafContainerT* neighborLeaf = *leafItr;
				if (leafVoxelMap.find(neighborLeaf) != leafVoxelMap.end() &&
						labeledLeafMapping.find(neighborLeaf) != labeledLeafMapping.end() && labeledLeafMapping[neighborLeaf] == label) {	// same supervoxel
					VData::Ptr neighborVoxel = leafVoxelMap[neighborLeaf];
					indicesToConsider.push_back(neighborVoxel->getCentroidCloudIndex());
				}
			}

			pcl::computePointNormal(centroidCloud, indicesToConsider, params, curvature);

			Eigen::Vector3f normal;
			normal[0] = params[0];
			normal[1] = params[1];
			normal[2] = params[2];

			voxel->setNormal(normal);

			// Normal Calculation end

			// check if SVMapping already contains the supervoxel
			if (SVMapping.find(label) != SVMapping.end()) {
				SVMapping[label]->getVoxelAVector()->push_back(voxel);
			} else {

				SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
				supervoxel->setLabel(label);
				supervoxel->getVoxelAVector()->push_back(voxel);

				// Add SV to SVMapping
				SVMapping.insert(pair<uint, typename SData::Ptr>(label, supervoxel));
			}

			// end if

		} else {
			leavesNotFoundWithSupervoxel++;
		}

	}

	cout << "scan 1 leaves without supervoxel: " << leavesNotFoundWithSupervoxel << endl;

	leafVoxelMap.clear();

	cout<<"Finding supervoxel normals " << endl;
	// calculating supervoxel normal
	SVMap::iterator svItr = SVMapping.begin();
	for (; svItr != SVMapping.end(); ++svItr) {

		int supervoxelPointCount = 0;
		SData::Ptr supervoxel = svItr->second;
		SData::VoxelVectorPtr voxels = supervoxel->getVoxelAVector();

		Eigen::Vector3f supervoxelNormal = Eigen::Vector3f::Zero();
		PointT supervoxelCentroid;

		double x(0), y(0), z(0), r(0), g(0), b(0);
		typename SData::VoxelVector::iterator voxelItr = voxels->begin();
		for (; voxelItr != voxels->end(); ++voxelItr) {

			int voxelSize = (*voxelItr)->getIndexVector()->size();
			supervoxelPointCount += voxelSize;

			supervoxelNormal += (*voxelItr)->getNormal();
			PointT p = (*voxelItr)->getCentroid();
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int voxelSize = voxels->size();
		if (voxelSize != 0) {
			supervoxelCentroid.x = x/voxelSize;
			supervoxelCentroid.y = y/voxelSize;
			supervoxelCentroid.z = z/voxelSize;
			supervoxelCentroid.r = r/voxelSize;
			supervoxelCentroid.g = g/voxelSize;
			supervoxelCentroid.b = b/voxelSize;

			supervoxel->setCentroidA(supervoxelCentroid);
		}

		supervoxel->setPointACount(supervoxelPointCount);

		if (!supervoxelNormal.isZero()) {
			supervoxelNormal.normalize();
			supervoxel->setNormal(supervoxelNormal);
		}

	}
}

void
createSuperVoxelMappingForScan2 (
		SVMap& SVMapping,
		const typename PointCloudT::Ptr scan,
		LabeledLeafMapT& labeledLeafMapping,
		//		const AdjacencyOctreeT adjTree2,
		const AdjacencyOctreeT adjTree1,
		KdTreeXYZ& svKdTree,
		std::vector<int>& treeLables) {

	AdjacencyOctreeT adjTree2;
	adjTree2.reset (new typename SupervoxelClusteringT::OctreeAdjacencyT(programOptions.vr));
	adjTree2->setInputCloud(scan);
	adjTree2->customBoundingBox(octreeBounds.minPt.x, octreeBounds.minPt.y, octreeBounds.minPt.z,
			octreeBounds.maxPt.x, octreeBounds.maxPt.y, octreeBounds.maxPt.z);
	adjTree2->addPointsFromInputCloud();

	cout << "Adjacency Octree Created for scan2 after transformation" << endl;

	SVMap::iterator svItr = SVMapping.begin();

	for (; svItr!=SVMapping.end(); ++svItr) {
		typename SData::Ptr supervoxel = svItr->second;
		supervoxel->clearScanBMapping();
		supervoxel->clearScanBData();
	}

	PointCloud<PointT>::iterator scanItr;
	int scanCounter = 0;

	LeafVoxelMapT leafVoxelMap;

	for (scanItr = scan->begin(); scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree2 -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree2 -> getLeafContainerAtPoint(a);

			if (leafVoxelMap.find(leaf) != leafVoxelMap.end()) {
				leafVoxelMap[leaf]->getIndexVector()->push_back(scanCounter);
			} else {
				VData::Ptr voxel = boost::shared_ptr<VData>(new VData());
				voxel->getIndexVector()->push_back(scanCounter);
				leafVoxelMap.insert(pair<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> (leaf, voxel));
			}

		} else {
			cout << "Not present in voxel: " << a.x << ' ' << a.y << ' ' << a.z << endl;
		}
	}

	// leafVoxelMap created for scan2

	LeafVoxelMapT::iterator leafVoxelItr;
	PointCloudT centroidCloud;
	int centroidCloudCounter = 0;

	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr, ++centroidCloudCounter) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first; // this is ocree2 leaf
		VData::Ptr voxel = leafVoxelItr->second;

		// compute Centroid
		typename VData::ScanIndexVectorPtr scanIndexVector = voxel->getIndexVector();
		typename VData::ScanIndexVector::iterator indexItr;

		PointT centroid;
		double x(0), y(0), z(0), r(0), g(0), b(0);
		for (indexItr = scanIndexVector->begin(); indexItr != scanIndexVector->end(); ++indexItr) {
			PointT p = scan->at(*indexItr);
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int size = scanIndexVector->size();
		centroid.x = x / size;
		centroid.y = y / size;
		centroid.z = z / size;
		centroid.r = r / size;
		centroid.g = g / size;
		centroid.g = b / size;

		centroidCloud.push_back(centroid);
		voxel->setCentroid(centroid);
		voxel->setCentroidCloudIndex(centroidCloudCounter);
	}

	// Setup search params
	int NN = SEARCH_SUPERVOXEL_NN;
	PointT voxelCentroid;
	PointXYZ queryPoint;
	std::vector<int> pointIdxNKNSearch(NN);
	std::vector<float> pointNKNSquaredDistance(NN);
	std::vector<int>::iterator intItr;


	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first; // octree 2 leaf
		VData::Ptr voxel = leafVoxelItr->second;
		typename SupervoxelClusteringT::LeafContainerT::const_iterator leafItr;


		bool searchForSupervoxel = false;
		bool presentInScan1 = adjTree1-> isVoxelOccupiedAtPoint(voxel->getCentroid());
		if (presentInScan1) {
			// scan1 leaf
			SupervoxelClusteringT::LeafContainerT* leaf1 = adjTree1->getLeafContainerAtPoint(voxel->getCentroid());

			if (labeledLeafMapping.find(leaf1) != labeledLeafMapping.end()) {

				unsigned int label = labeledLeafMapping[leaf1];

				// check if SVMapping already contains the supervoxel
				if (SVMapping.find(label) != SVMapping.end()) {
					SData::Ptr supervoxel = SVMapping[label];
					supervoxel->getVoxelBVector()->push_back(voxel);
				} else {
					SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
					supervoxel->setLabel(label);
					supervoxel->getVoxelBVector()->push_back(voxel);
					// Add SV to SVMapping
					SVMapping.insert(pair<uint, typename SData::Ptr>(label, supervoxel));
				}

			} else {
				// leaf exists in scan 1 but doesn't have a supervoxel label
				searchForSupervoxel = true;
			}

		} else {
			// search for neareset supervoxel for this leaf
			searchForSupervoxel = true;
		}

		if (searchForSupervoxel) {

			// search for the nearest supervoxel with the modified distance function which takes normal into account

			// calculate normal for this leaf
			Eigen::Vector4f params = Eigen::Vector4f::Zero();
			float curvature;
			std::vector<int> indicesToConsider;

			for (leafItr = leaf->cbegin(); leafItr != leaf->cend(); ++leafItr) {
				typename SupervoxelClusteringT::LeafContainerT* neighborLeaf = *leafItr;
				if (leafVoxelMap.find(neighborLeaf) != leafVoxelMap.end()) {
					VData::Ptr neighborVoxel = leafVoxelMap[neighborLeaf];
					indicesToConsider.push_back(neighborVoxel->getCentroidCloudIndex());
				}
			}

			pcl::computePointNormal(centroidCloud, indicesToConsider, params, curvature);

			Eigen::Vector3f normal;
			normal[0] = params[0];
			normal[1] = params[1];
			normal[2] = params[2];

			if (!normal.isZero()) {
				normal.normalize();
				voxel->setNormal(normal);
			}
			// Normal Calculation end

			// search supervoxels for the scan 2 leaf
			voxelCentroid = voxel->getCentroid();

			// Clear prev indices
			pointIdxNKNSearch.clear();
			pointNKNSquaredDistance.clear();

			queryPoint.x = voxelCentroid.x;
			queryPoint.y = voxelCentroid.y;
			queryPoint.z = voxelCentroid.z;

			// search for NN centroids

			if (svKdTree.nearestKSearch (queryPoint,NN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
			{

				SData::Ptr supervoxel;
				Eigen::Vector3f supervoxelNormal;

				int closestSupervoxelLabel = -1;
				double minDistance = INT_MAX;
				for (intItr = pointIdxNKNSearch.begin(); intItr != pointIdxNKNSearch.end(); ++intItr) {

					int index = *intItr;

					float euclideanD = sqrt(pointNKNSquaredDistance[index]);
					int svLabel = treeLables[index];
					supervoxel = SVMapping[svLabel];
					supervoxelNormal = supervoxel->getNormal();

					double d = acos(normal.dot(supervoxelNormal)) * 2 / M_PI;
					d = 1- log2(1.0 - d);

					d *= euclideanD;
					if (d < minDistance) {
						d = minDistance;
						closestSupervoxelLabel = svLabel;
					}

				}

				if (closestSupervoxelLabel > 0)
					SVMapping[closestSupervoxelLabel]->getVoxelBVector()->push_back(voxel);

			}

			// search for matching normal among the NN supervoxels using new Distance Function:
			// D = (1 - log2( 1 -  acos(SupervoxelNormal dot VoxelNormal) / (Pi/2) )) * Euclidean_Distance
		}
	}

	// end supervoxel iteration
	leafVoxelMap.clear();
}

void
calculateSupervoxelScanBData(SVMap& supervoxelMapping, PointCloudT::Ptr scanB) {

	typename SVMap::iterator svItr;
	typename SData::VoxelVector::iterator voxelItr;
	typename VData::ScanIndexVector::iterator indexItr;

	// calculate centroids for scan B and count of B points
	for (svItr = supervoxelMapping.begin(); svItr != supervoxelMapping.end(); ++svItr) {

		int supervoxelPointCount = 0;
		SData::Ptr supervoxel = svItr->second;
		SData::VoxelVectorPtr voxels = supervoxel->getVoxelBVector();

		PointT supervoxelCentroid;

		double x(0), y(0), z(0), r(0), g(0), b(0);

		for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {

			VData::Ptr voxel = *voxelItr;
			typename VData::ScanIndexVectorPtr indexVector = voxel->getIndexVector();
			PointT voxelCentroid;

			double xv(0), yv(0), zv(0), rv(0), gv(0), bv(0);
			for (indexItr = indexVector->begin(); indexItr != indexVector->end(); ++indexItr) {
				PointT pv = scanB->at(*indexItr);

				xv += pv.x;
				yv += pv.y;
				zv += pv.z;
				rv += pv.r;
				gv += pv.g;
				bv += pv.b;

			}

			int numberOfPoints = indexVector->size();

			if (numberOfPoints != 0) {
				voxelCentroid.x = xv/numberOfPoints;
				voxelCentroid.y = yv/numberOfPoints;
				voxelCentroid.z = zv/numberOfPoints;
				voxelCentroid.r = rv/numberOfPoints;
				voxelCentroid.g = gv/numberOfPoints;
				voxelCentroid.b = bv/numberOfPoints;

				voxel->setCentroid(voxelCentroid);
			}

			supervoxelPointCount += numberOfPoints;

			PointT p = (*voxelItr)->getCentroid();
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int voxelsSize = voxels->size();
		if (voxelsSize != 0) {
			supervoxelCentroid.x = x/voxelsSize;
			supervoxelCentroid.y = y/voxelsSize;
			supervoxelCentroid.z = z/voxelsSize;
			supervoxelCentroid.r = r/voxelsSize;
			supervoxelCentroid.g = g/voxelsSize;
			supervoxelCentroid.b = b/voxelsSize;

			supervoxel->setCentroidB(supervoxelCentroid);
		}

		supervoxel->setPointBCount(supervoxelPointCount);
	}
}

void
createKDTreeForSupervoxels(
		SVMap& supervoxelMap,
		KdTreeXYZ& svKdTree,
		std::vector<int>& labels) {

	PointCloudXYZ::Ptr svLabelCloud = boost::shared_ptr<PointCloudXYZ>(new PointCloudXYZ());

	SVMap::iterator svItr;

	PointT supervoxelCentroid;
	PointXYZ treePoint;

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		SData::Ptr supervoxel = svItr->second;
		int label = svItr->first;

		supervoxelCentroid = supervoxel->getCentroidA();

		treePoint.x = supervoxelCentroid.x;
		treePoint.y = supervoxelCentroid.y;
		treePoint.z = supervoxelCentroid.z;
		labels.push_back(label);

		svLabelCloud->push_back(treePoint);
	}

	svKdTree.setInputCloud(svLabelCloud);
}

/*
 *	Function calculates the Mutual Information between ScanA and transformed ScanB
 *	Mutual Information Steps (Only Normal Feature for now):
 *
 *	A. Feature space -> Normal
 *
 *		1. Find The normals for scanB in the supervoxels
 *		2. Find Normal Vector Code for scan B in all supervoxels
 *		3. Calculate H(X) for the supervoxels on the basis of Normal Vector Codes for Scan A
 *		4. Calculate H(Y) for the supervoxels on the basis of Normal Vector Codes for Scan B
 *		5. Calculate H(X, Y) for the supervoxels on the basis of Normal Vector Codes for both Scan A and Scan B (Joint Histo)
 *		6. Calculate MI(X,Y) = H(X) + H (Y) - H(X,Y)
 *		7. return as the value of the function -MI(X,Y)
 *
 *	Simple MI not working
 *	Attempts
 *
 *		1. Normalizing Mutual Information on the basis of points present in supervoxel.
 */

double
calculateMutualInformation(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	// Feature 1
	map<int, double> variancex_XProbability;
	map<int, double> variancex_YProbability;
	map<string, double> variancex_XYProbability;

	// Feature 2
	map<int, double> variancey_XProbability;
	map<int, double> variancey_YProbability;
	map<string, double> variancey_XYProbability;

	// Feature 3
	map<int, double> variancez_XProbability;
	map<int, double> variancez_YProbability;
	map<string, double> variancez_XYProbability;

	// Feature 4 = centroid resultant
	map<int, double> centroid_XProbability;
	map<int, double> centroid_YProbability;
	map<string, double> centroid_XYProbability;

	unsigned int totalAPointsInOverlappingRegion(0), totalBPointsInOverlappingRegion(0), size(0);

	SVMap::iterator svItr;

	for (svItr = SVMapping.begin(); svItr!=SVMapping.end(); ++svItr) {

		// Write MI Code
		int svLabel = svItr->first;
		typename SData::Ptr supervoxel = svItr->second;

		unsigned int counterA = supervoxel->getPointACount();
		unsigned int counterB = supervoxel->getPointBCount();

		if (counterA > MIN_POINTS_IN_SUPERVOXEL && counterB > MIN_POINTS_IN_SUPERVOXEL) {

			// Variance Calculate

			double var_x_A(0), var_x_B(0);
			double var_y_A(0), var_y_B(0);
			double var_z_A(0), var_z_B(0);
			double centroidNorm_A(0), centroidNorm_B(0);

			int varx_ACode = supervoxel->getVarianceXCodeA();
			int vary_ACode = supervoxel->getVarianceYCodeA();
			int varz_ACode = supervoxel->getVarianceZCodeA();
			int centroid_ACode = supervoxel->getCentroidCodeA();

			bool calculateAVariance = false;
			if (varx_ACode == 0 || vary_ACode == 0 || varz_ACode == 0 || centroid_ACode == 0)
				calculateAVariance = true;

			SData::VoxelVectorPtr voxelsA = supervoxel->getVoxelAVector();
			SData::VoxelVectorPtr voxelsB = supervoxel->getVoxelBVector();

			PointT centroidA = supervoxel->getCentroidA();
			PointT centroidB = supervoxel->getCentroidB();

//			cout << "Label: " << svLabel << endl;
//			cout << "A: " << centroidA.x << ' ' << centroidA.y << ' ' << centroidA.z << endl;
//			cout << "B: " << centroidB.x << ' ' << centroidB.y << ' ' << centroidB.z << endl;

			SData::VoxelVector::iterator voxelItr;
			VData::ScanIndexVectorPtr indexVector;
			VData::ScanIndexVector::iterator indexVectorItr;

			bool debug = false;
			if (svLabel == 678) {
//				debug = true;
			}

			if (calculateAVariance) {

				// calculate A variance -> only once
				for (voxelItr = voxelsA->begin(); voxelItr != voxelsA->end(); ++voxelItr) {

					VData::Ptr voxel = *voxelItr;
					indexVector = voxel->getIndexVector();

					if (debug)
						cout << "Points " << endl;

					for (indexVectorItr = indexVector->begin(); indexVectorItr != indexVector->end(); ++indexVectorItr) {

						double x = scan1->at(*indexVectorItr).x;
						double y = scan1->at(*indexVectorItr).y;
						double z = scan1->at(*indexVectorItr).z;

						if (debug)
							cout << "x: " << x << " y: " << y << " z: " << z << endl;

						var_x_A += square<double> (x-centroidA.x);
						var_y_A += square<double> (y-centroidA.y);
						var_z_A += square<double> (z-centroidA.z);
					}


				}

				var_x_A /= counterA;
				var_y_A /= counterA;
				var_z_A /= counterA;
				centroidNorm_A = sqrt((square<double>(centroidA.x) +
								square<double>(centroidA.y) +
								square<double>(centroidA.z)));

				if (debug) {
					cout << "VarianceX in A : " << var_x_A << endl;
					cout << "VarianceY in A : " << var_y_A << endl;
					cout << "VarianceZ in A : " << var_z_A << endl;
				}
				centroid_ACode = getCentroidResultantCode(centroidNorm_A);
				varx_ACode = getVarianceCode(var_x_A);
				vary_ACode = getVarianceCode(var_y_A);
				varz_ACode = getVarianceCode(var_z_A);

				supervoxel->setCentroidCodeA(centroid_ACode);
				supervoxel->setVarianceXCodeA(varx_ACode);
				supervoxel->setVarianceYCodeA(vary_ACode);
				supervoxel->setVarianceZCodeA(varz_ACode);
			}

			// calculate B variance
			for (voxelItr = voxelsB->begin(); voxelItr != voxelsB->end(); ++ voxelItr) {

				VData::Ptr voxel = *voxelItr;
				indexVector = voxel->getIndexVector();

				for (indexVectorItr = indexVector->begin(); indexVectorItr != indexVector->end(); ++indexVectorItr) {

					double x = scan2->at(*indexVectorItr).x;
					double y = scan2->at(*indexVectorItr).y;
					double z = scan2->at(*indexVectorItr).z;

					var_x_B += square<double> (x-centroidB.x);
					var_y_B += square<double> (y-centroidB.y);
					var_z_B += square<double> (z-centroidB.z);
				}

			}

			var_x_B /= counterB;
			var_y_B /= counterB;
			var_z_B /= counterB;
			centroidNorm_B = sqrt((square<double>(centroidB.x) +
											square<double>(centroidB.y) +
											square<double>(centroidB.z)));

			int centroid_BCode = getCentroidResultantCode(centroidNorm_B);
			int varx_BCode = getVarianceCode(var_x_B);
			int vary_BCode = getVarianceCode(var_y_B);
			int varz_BCode = getVarianceCode(var_z_B);

			supervoxel->setCentroidCodeB(centroid_BCode);
			supervoxel->setVarianceXCodeB(varx_BCode);
			supervoxel->setVarianceYCodeB(vary_BCode);
			supervoxel->setVarianceZCodeB(varz_BCode);

			string centroid_ABCode = boost::str(boost::format("%d_%d")%centroid_ACode%centroid_BCode);
			string varx_ABCode = boost::str(boost::format("%d_%d")%varx_ACode%varx_BCode);
			string vary_ABCode = boost::str(boost::format("%d_%d")%vary_ACode%vary_BCode);
			string varz_ABCode = boost::str(boost::format("%d_%d")%varz_ACode%varz_BCode);

			supervoxel->setCentroidCodeAB(centroid_ABCode);
			supervoxel->setVarianceXCodeAB(varx_ABCode);
			supervoxel->setVarianceYCodeAB(vary_ABCode);
			supervoxel->setVarianceZCodeAB(varz_ABCode);

			// Variance X Features

			if (centroid_XProbability.find(centroid_ACode) != centroid_XProbability.end()) {
				centroid_XProbability[centroid_ACode] += 1;
			}  else {
				centroid_XProbability.insert(pair<int, double> (centroid_ACode, 1.0));
			}

			if (variancex_XProbability.find(varx_ACode) != variancex_XProbability.end()) {
				variancex_XProbability[varx_ACode] += 1;
			}  else {
				variancex_XProbability.insert(pair<int, double> (varx_ACode, 1.0));
			}

			if (variancey_XProbability.find(vary_ACode) != variancey_XProbability.end()) {
				variancey_XProbability[vary_ACode] += 1;
			}  else {
				variancey_XProbability.insert(pair<int, double> (vary_ACode, 1.0));
			}

			if (variancez_XProbability.find(varz_ACode) != variancez_XProbability.end()) {
				variancez_XProbability[varz_ACode] += 1;
			}  else {
				variancez_XProbability.insert(pair<int, double> (varz_ACode, 1.0));
			}

			// Variance Y Features

			if (centroid_YProbability.find(centroid_BCode) != centroid_YProbability.end()) {
				centroid_YProbability[centroid_BCode] += 1;
			}  else {
				centroid_YProbability.insert(pair<int, double> (centroid_BCode, 1.0));
			}

			if (variancex_YProbability.find(varx_BCode) != variancex_YProbability.end()) {
				variancex_YProbability[varx_BCode] += 1;
			}  else {
				variancex_YProbability.insert(pair<int, double> (varx_BCode, 1.0));
			}

			if (variancey_YProbability.find(vary_BCode) != variancey_YProbability.end()) {
				variancey_YProbability[vary_BCode] += 1;
			}  else {
				variancey_YProbability.insert(pair<int, double> (vary_BCode, 1.0));
			}

			if (variancez_YProbability.find(varz_BCode) != variancez_YProbability.end()) {
				variancez_YProbability[varz_BCode] += 1;
			}  else {
				variancez_YProbability.insert(pair<int, double> (varz_BCode, 1.0));
			}

			// Variance XY Features

			if (centroid_XYProbability.find(centroid_ABCode) != centroid_XYProbability.end()) {
				centroid_XYProbability[centroid_ABCode] += 1;
			}  else {
				centroid_XYProbability.insert(pair<string, double> (centroid_ABCode, 1.0));
			}

			if (variancex_XYProbability.find(varx_ABCode) != variancex_XYProbability.end()) {
				variancex_XYProbability[varx_ABCode] += 1;
			}  else {
				variancex_XYProbability.insert(pair<string, double> (varx_ABCode, 1.0));
			}

			if (variancey_XYProbability.find(vary_ABCode) != variancey_XYProbability.end()) {
				variancey_XYProbability[vary_ABCode] += 1;
			}  else {
				variancey_XYProbability.insert(pair<string, double> (vary_ABCode, 1.0));
			}

			if (variancez_XYProbability.find(varz_ABCode) != variancez_XYProbability.end()) {
				variancez_XYProbability[varz_ABCode] += 1;
			}  else {
				variancez_XYProbability.insert(pair<string, double> (varz_ABCode, 1.0));
			}

			// End Variance computation

			totalAPointsInOverlappingRegion += counterA;
			totalBPointsInOverlappingRegion += counterB;
			size++;

		}
	}

	// Calculating probabilities for all norm codes
	map<int, double>::iterator itr;

	// Calculating prob for all events of X for features x,y,z

	for (itr = centroid_XProbability.begin(); itr != centroid_XProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}


	for (itr = variancex_XProbability.begin(); itr != variancex_XProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	for (itr = variancey_XProbability.begin(); itr != variancey_XProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	for (itr = variancez_XProbability.begin(); itr != variancez_XProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	// Calculating prob for all events of Y for feeatures x,y,z

	for (itr = centroid_YProbability.begin(); itr != centroid_YProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	for (itr = variancex_YProbability.begin(); itr != variancex_YProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	for (itr = variancey_YProbability.begin(); itr != variancey_YProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	for (itr = variancez_YProbability.begin(); itr != variancez_YProbability.end(); ++itr) {
		double x = ((double)itr->second) / size;
		itr->second = x;
	}

	map<string, double>::iterator xyItr;

	// Calculating prob for all events of XY for features x,y,z

	for (xyItr = centroid_XYProbability.begin(); xyItr != centroid_XYProbability.end(); ++xyItr) {
		double xy = ((double)xyItr->second) / size;
		xyItr->second = xy;
	}

	for (xyItr = variancex_XYProbability.begin(); xyItr != variancex_XYProbability.end(); ++xyItr) {
		double xy = ((double)xyItr->second) / size;
		xyItr->second = xy;
	}

	for (xyItr = variancey_XYProbability.begin(); xyItr != variancey_XYProbability.end(); ++xyItr) {
		double xy = ((double)xyItr->second) / size;
		xyItr->second = xy;
	}

	for (xyItr = variancez_XYProbability.begin(); xyItr != variancez_XYProbability.end(); ++xyItr) {
		double xy = ((double)xyItr->second) / size;
		xyItr->second = xy;
	}

	// Probability calculation complete

	// calculate MI for overlapping supervoxels using normalXProbability, randomY and normalXYProbability

	double hX(0), hY(0), hXY(0);

	for (svItr = SVMapping.begin(); svItr != SVMapping.end(); ++svItr) {

		SData::Ptr supervoxel = svItr->second;

		unsigned int counterA = supervoxel->getPointACount();
		unsigned int counterB = supervoxel->getPointBCount();

		if (counterA > MIN_POINTS_IN_SUPERVOXEL && counterB > MIN_POINTS_IN_SUPERVOXEL) {

			// MI calculation using varX, varY, varZ as features

			int centroidACode = supervoxel->getCentroidCodeA();
			int centroidBCode = supervoxel->getCentroidCodeB();
			string centroidABCode = supervoxel->getCentroidCodeAB();

			int varxACode = supervoxel->getVarianceXCodeA();
			int varxBCode = supervoxel->getVarianceXCodeB();
			string varxABCode = supervoxel->getVarianceXCodeAB();

			int varyACode = supervoxel->getVarianceYCodeA();
			int varyBCode = supervoxel->getVarianceYCodeB();
			string varyABCode = supervoxel->getVarianceYCodeAB();

			int varzACode = supervoxel->getVarianceZCodeA();
			int varzBCode = supervoxel->getVarianceZCodeB();
			string varzABCode = supervoxel->getVarianceZCodeAB();

			double centroidAPro = centroid_XProbability.at(centroidACode);
			double varxAPro = variancex_XProbability.at(varxACode);
			double varyAPro = variancey_XProbability.at(varyACode);
			double varzAPro = variancez_XProbability.at(varzACode);

			double centroidBPro = centroid_YProbability.at(centroidBCode);
			double varxBPro = variancex_YProbability.at(varxBCode);
			double varyBPro = variancey_YProbability.at(varyBCode);
			double varzBPro = variancez_YProbability.at(varzBCode);

			double centroidABPro = centroid_XYProbability.at(centroidABCode);
			double varxABPro = variancex_XYProbability.at(varxABCode);
			double varyABPro = variancey_XYProbability.at(varyABCode);
			double varzABPro = variancez_XYProbability.at(varzABCode);

			hX += varxAPro * log(varxAPro) + varyAPro * log(varyAPro) + varzAPro * log(varzAPro) + centroidAPro * log(centroidAPro);
			hY += varxBPro * log(varxBPro) + varyBPro * log(varyBPro) + varzBPro * log(varzBPro) + centroidBPro * log(centroidBPro);
			hXY += varxABPro * log(varxABPro) + varyABPro * log(varyABPro) + varzABPro * log(varzABPro) + centroidABPro * log(centroidABPro);

//			hX += centroidAPro * log(centroidAPro);
//			hY += centroidBPro * log(centroidBPro);
//			hXY += centroidABPro * log(centroidABPro);

		}

	}

	hX *= -1;
	hY *= -1;
	hXY *= -1;

//	cout << hX << ' ' << hY << ' ' << hXY << endl;
	double mi = hX + hY - hXY;
	double nmi = (hX + hY) / hXY;

	return mi;
}

struct MI_Opti_Data{
	SVMap* svMap;
	PointCloudT::Ptr scan1;
	PointCloudT::Ptr scan2;
};

/*
 *	Optimization Function
 * 	Note: Scan A Data remains constant including the supervoxels generated
 * 	v is the transformation vector XYZRPY best guess till now
 * 	params will include :-
 *
 * 	1. SVMap -> Supervoxel to scanA Indices will remain constant
 * 	2. SVMap -> Supervoxel to scanB Indices (will change in every iteration)
 * 	3. LeafMapContainer -> This will always be constant (Will be used to find supervoxels for scan B points)
 * 	4. Octree -> To find the leaf container for scan B points
 * 	5. Scan A
 * 	6. Scan B
 *
 *	Function steps:
 *
 *	1. Transform B with current XYZRPY
 *	2. Find the corresponding Supervoxels
 *	3. Find the supervoxels with a minimum number of points
 *	4. Apply Mutual Information on the common Supervoxels
 *
 *	CommonSupervoxels will be used for MI calculation
 *
 */

double mi_f (const gsl_vector *pose, void* params) {

	// Initialize All Data
	double x, y, z, roll, pitch ,yaw;
	x = gsl_vector_get(pose, 0);
	y = gsl_vector_get(pose, 1);
	z = gsl_vector_get(pose, 2);
	roll = gsl_vector_get(pose, 3);
	pitch = gsl_vector_get(pose, 4);
	yaw = gsl_vector_get(pose, 5);

	MI_Opti_Data* miOptiData = (MI_Opti_Data*) params;

	PointCloudT::Ptr scan1 = miOptiData->scan1;
	PointCloudT::Ptr scan2 = miOptiData->scan2;
	PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<PointCloudT>(new PointCloudT());

	SVMap* SVMapping = miOptiData->svMap;

	// Create Transformation
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
	transform.translation() << x,y,z;
	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

	// Transform point cloud
	pcl::transformPointCloud(*scan2, *transformedScan2, transform);

	// Clear SVMap for new scan2 properties

	SVMap::iterator svItr;
	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
		SData::Ptr supervoxel = svItr->second;
		supervoxel->clearScanBData();
	}

	calculateSupervoxelScanBData(*SVMapping, transformedScan2);

	double mi = calculateMutualInformation(*SVMapping, scan1, transformedScan2);

	//	cout << "MI Function Called with refreshed values" << mi << endl;

	return -mi;
}

Eigen::Affine3d optimize(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, gsl_vector* baseX) {

	MI_Opti_Data* mod = new MI_Opti_Data();
	mod->scan1 = scan1;
	mod->scan2 = scan2;
	mod->svMap = &SVMapping;

	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;

	//	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
	gsl_multimin_fminimizer *s = NULL;

	gsl_vector *ss;
	gsl_multimin_function minex_func;

	size_t iter = 0;
	int status;
	double size;

	/* Set  initial step sizes to 1 */
	ss = gsl_vector_alloc (6);
	gsl_vector_set (ss, 0, 0.2);
	gsl_vector_set (ss, 1, 0.2);
	gsl_vector_set (ss, 2, 0.2);
	gsl_vector_set (ss, 3, 0.1);
	gsl_vector_set (ss, 4, 0.1);
	gsl_vector_set (ss, 5, 0.1);

	/* Initialize method and iterate */
	minex_func.n = 6; // Dimension
	minex_func.f = mi_f;
	minex_func.params = mod;

	s = gsl_multimin_fminimizer_alloc (T, 6);
	gsl_multimin_fminimizer_set (s, &minex_func, baseX, ss);

	do {

		iter++;
		status = gsl_multimin_fminimizer_iterate(s);

		if (status)
			break;

		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 2e-2);

		printf("%5d f() = %7.3f size = %.3f x=%f y=%f z=%f roll=%f pitch=%f yaw=%f \n",
				(int)iter,
				s->fval,
				size, gsl_vector_get (s->x, 0), gsl_vector_get (s->x, 1), gsl_vector_get (s->x, 2),
				gsl_vector_get (s->x, 3), gsl_vector_get (s->x, 4), gsl_vector_get (s->x, 5));

		if (status == GSL_SUCCESS || iter >= 100) {

			cout << "MI= " << s->fval << " Iteration: " << iter << endl;

			cout << "Base Transformation: " << endl;

			double tx = gsl_vector_get (baseX, 0);
			double ty = gsl_vector_get (baseX, 1);
			double tz = gsl_vector_get (baseX, 2);
			double roll = gsl_vector_get (baseX, 3);
			double pitch = gsl_vector_get (baseX, 4);
			double yaw = gsl_vector_get (baseX, 5);

			cout << "Tx: " << tx << endl;
			cout << "Ty: " << ty << endl;
			cout << "Tz: " << tz << endl;
			cout << "Roll: " << roll << endl;
			cout << "Pitch: " << pitch << endl;
			cout << "Yaw: " << yaw << endl;


			cout << "Converged to minimum at " << endl;

			tx = gsl_vector_get (s->x, 0);
			ty = gsl_vector_get (s->x, 1);
			tz = gsl_vector_get (s->x, 2);
			roll = gsl_vector_get (s->x, 3);
			pitch = gsl_vector_get (s->x, 4);
			yaw = gsl_vector_get (s->x, 5);

			cout << "Tx: " << tx << endl;
			cout << "Ty: " << ty << endl;
			cout << "Tz: " << tz << endl;
			cout << "Roll: " << roll << endl;
			cout << "Pitch: " << pitch << endl;
			cout << "Yaw: " << yaw << endl;

			Eigen::Affine3d resultantTransform = Eigen::Affine3d::Identity();
			resultantTransform.translation() << tx, ty, tz;
			resultantTransform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
			resultantTransform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
			resultantTransform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

			gsl_vector_free(ss);
			gsl_multimin_fminimizer_free(s);

			return resultantTransform;
		}

	} while(status == GSL_CONTINUE && iter < 100);

	//	gsl_vector_free(baseX);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);

	return Eigen::Affine3d::Identity();
}

/*
 * 	Returns a unique Normal Vector code for a group of normalized vectors in the range
 * 	[x,y,z] - [x+dx, y+dy, z+dy]
 *
 * 	NormalVectorCode is a label
 * 	Its value is not of any significance
 *
 *	-1.0 -> 0
 *	-0.8 -> 1
 *	-0.6 -> 2
 *	and so on
 *
 */

Eigen::Vector4f
getNormalizedVectorCode(Eigen::Vector3f vector) {

	Eigen::Vector4f resultantCode = Eigen::Vector4f::Zero();

	float x = vector[0];
	float y = vector[1];
	float z = vector[2];

	if (abs(x) > 1 || abs(y) > 1 || abs(z) > 1)
		return resultantCode;

	int a(0), b(0), c(0);
	float dx(NORM_DX), dy(NORM_DY), dz(NORM_DZ), diff;

	int Dx(0),Dy(0),Dz(0);
	int Tx(0), Ty(0), Tz(0);

	Dx = 1.0 / dx;
	Dy = 1.0 / dy;
	Dz = 1.0 / dz;

	Tx = 2*Dx + 1;
	Ty = 2*Dy + 1;
	Tz = 2*Dz + 1;

	while (dx < abs(x)) {
		dx += NORM_DX;
		a++;
	}

	//	diff = dx - abs(x);
	//	if (2 * diff > NORM_DX) // Moving to a closer step
	//		a++;


	while (dy < abs(y)) {
		dy += NORM_DY;
		b++;
	}

	//	diff = dy - abs(y);
	//	if (2*diff > NORM_DY)
	//		b++;

	while (dz < abs(z)) {
		dz += NORM_DZ;
		c++;
	}

	//	diff = dz - abs(z);
	//	if (2*diff > NORM_DZ)
	//		c++;

	if (x >= 0)
		a = Dx + a;
	else
		a = Dx - a;

	if (y >= 0)
		b = Dy + b;
	else
		b = Dy - b;

	if (z >= 0)
		c = Dz + c;
	else
		c = Dx - c;

	int code = a + b * Tx + c * (Tx * Ty);

	resultantCode[0] = a;
	resultantCode[1] = b;
	resultantCode[2] = c;
	resultantCode[3] = code;

	return resultantCode;
}

/*
 *
 */
int
getCentroidResultantCode(double norm) {

	int a(0);
	double dR = NORM_R;
	double dNorm = dR;

	while (dNorm < norm) {
		a++;
		dNorm += dR;
	}

	double diff = dNorm - norm;
	diff *= 2;

	if (diff - dR < 0)
		a++;

//	cout << "Norm: " << norm << ' ' << " Result: " << a << endl;

	return a;
}

int
getVarianceCode(double norm) {

	int a(0);
	double dR = NORM_VR;
	double dNorm = dR;

	while (dNorm < norm) {
		a++;
		dNorm += dR;
	}

	double diff = dNorm - norm;
	diff *= 2;

	if (diff - dR < 0)
		a++;

//	cout << "Norm: " << norm << ' ' << " Result: " << a << endl;

	return a;
}


void
showPointClouds(PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, string viewerTitle) {

	boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer (viewerTitle));
	viewer->setBackgroundColor (0,0,0);

	string id1("scan1"), id2("scan2");

	visualization::PointCloudColorHandlerRGBField<PointT> rgb1(scan1);
	viewer->addPointCloud<PointT> (scan1, rgb1, id1);

	visualization::PointCloudColorHandlerRGBField<PointT> rgb2(scan2);
	viewer->addPointCloud<PointT> (scan2, rgb2, id2);

	viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, id2);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}

void
showPointCloud(typename PointCloudT::Ptr scan) {

	boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer ("Supervoxel Based MI Viewer"));
	viewer->setBackgroundColor (0,0,0);

	string id1("scan");

	visualization::PointCloudColorHandlerRGBField<PointT> rgb1(scan);
	viewer->addPointCloud<PointT> (scan, rgb1, id1);

	viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}

void
showTestSuperVoxel(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	// Display all supervoxels with count A and count B

	SVMap::iterator svItr;
	SData::VoxelVector::iterator voxelItr;
	VData::ScanIndexVector::iterator indexItr;

	int SV = programOptions.test;
	typename PointCloudT::Ptr newCloud (new PointCloudT);

	for (svItr = SVMapping.begin(); svItr!=SVMapping.end(); ++svItr) {

		// Write MI Code
		int svLabel = svItr->first;
		typename SData::Ptr supervoxel = svItr->second;
		bool showSupervoxel = false;

		if (svLabel == SV)
			showSupervoxel = true;

		SData::VoxelVectorPtr voxelsA = supervoxel->getVoxelAVector();
		SData::VoxelVectorPtr voxelsB = supervoxel->getVoxelBVector();

		int counterA(0), counterB(0);
		for (voxelItr = voxelsA->begin(); voxelItr != voxelsA->end(); ++ voxelItr) {
			VData::Ptr voxel = (*voxelItr);
			counterA+= voxel->getIndexVector()->size();

			if (showSupervoxel) {

				for (indexItr = voxel->getIndexVector()->begin(); indexItr != voxel->getIndexVector()->end(); ++indexItr) {

					PointT p = scan1->at(*indexItr);

					p.r = 255;
					p.g = 0;
					p.b = 0;

					newCloud->push_back(p);
				}


			}

		}

		for (voxelItr = voxelsB->begin(); voxelItr != voxelsB->end(); ++ voxelItr) {
			VData::Ptr voxel = (*voxelItr);
			counterB += voxel->getIndexVector()->size();

			if (showSupervoxel) {

				for (indexItr = voxel->getIndexVector()->begin(); indexItr != voxel->getIndexVector()->end(); ++indexItr) {

					PointT p = scan2->at(*indexItr);

					p.r = 0;
					p.g = 255;
					p.b = 0;

					newCloud->push_back(p);
				}


			}
		}

		cout << svLabel << '\t' << "A: " << counterA << '\t' << "B: " << counterB << endl;
	}

	showPointCloud(newCloud);
}

