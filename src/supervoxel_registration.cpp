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


#include <cmath>
#include <iomanip>

#include "supervoxel_octree_pointcloud_adjacency.h"
#include "supervoxel_mapping.hpp"
#include "supervoxel_registration.h"
#include "supervoxel_optimize.h"
#include "supervoxel_util.hpp"


using namespace svr;

SupervoxelRegistration::SupervoxelRegistration(float voxelR, float seedR):
	supervoxelClustering (voxelR, seedR),
	vr (voxelR),
	sr (seedR),
	octree_bounds_()
{

	debug = false;
	octree_bounds_.minPt.x = -120;
	octree_bounds_.minPt.y = -120;
	octree_bounds_.minPt.z = -20;

	octree_bounds_.maxPt.x = 120;
	octree_bounds_.maxPt.y = 120;
	octree_bounds_.maxPt.z = 20;
}

SupervoxelRegistration::~SupervoxelRegistration() {

}

void
SupervoxelRegistration::setScans(PointCloudT::Ptr scanA, PointCloudT::Ptr scanB) {
	this->A = scanA;
	this->B = scanB;
}

void
SupervoxelRegistration::prepareForRegistration() {
		std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxelClusters = initializeVoxels();
		std::cout << "Number of supervoxels: " << supervoxelMap.size() << std::endl;
		createSuperVoxelMappingForScan1();
		createKDTreeForSupervoxels();
}

// Return trans
Eigen::Matrix4d
SupervoxelRegistration::alignScans() {

	prepareForRegistration();

	cout << "Debug Mode: Preparation Complete" << endl;

	Eigen::Affine3d trans_last = Eigen::Affine3d::Identity();
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

		transformPointCloud (*B, *transformedScan2, trans_last);
		createSuperVoxelMappingForScan2(transformedScan2);

		cout << "Debug Mode: Printing supervoxel Map..." << endl;

		printSupervoxelMap();

		cout << "Iteration " << iteration+1 << " ..." << endl;

		svr_optimize::svr_opti_data opti_data;
//		opti_data.scan1 = A;
		opti_data.scan2 = B;
		opti_data.svMap = &supervoxelMap;
		opti_data.t = trans_last;

		trans_new = optimize(opti_data);

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

	Eigen::Matrix4d result = trans_new.inverse().matrix();
	cout << "Resultant Transformation: " << endl << result;
	cout << endl;

	return result;

}

/*
 * Initializes supevoxelCLustering for scan A and labeledLeafMap containing leaf supervoxel mapping
 */
std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr>
SupervoxelRegistration::initializeVoxels() {

	if (octree_bounds_.minPt.x == 0 && octree_bounds_.maxPt.x == 0) {
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
	}

	supervoxelClustering.setVoxelResolution(vr);
	supervoxelClustering.setSeedResolution(sr);
	supervoxelClustering.setInputCloud(A);
	supervoxelClustering.setColorImportance(0);
	supervoxelClustering.setSpatialImportance(1);
	supervoxelClustering.setNormalImportance(1);

	supervoxelClustering.getOctreeeAdjacency()->customBoundingBox(octree_bounds_.minPt.x, octree_bounds_.minPt.y, octree_bounds_.minPt.z,
			octree_bounds_.maxPt.x, octree_bounds_.maxPt.y, octree_bounds_.maxPt.z);

	// Not being used for now
	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxelClusters;
	supervoxelClustering.extract(supervoxelClusters);
	supervoxelClustering.getLabeledLeafContainerMap(leafMapping);
	return supervoxelClusters;
}


void
SupervoxelRegistration::createSuperVoxelMappingForScan1 () {

	AdjacencyOctreeT adjTree = supervoxelClustering.getOctreeeAdjacency();

	pcl::PointCloud<PointT>::iterator scanItr;
	int scanCounter = 0;

	LeafVoxelMapT leafVoxelMap;

	for (scanItr = A->begin(); scanItr != A->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			if (leafVoxelMap.find(leaf) != leafVoxelMap.end()) {
				leafVoxelMap[leaf]->getIndexVector()->push_back(scanCounter);
			} else {
				VData::Ptr voxel = boost::shared_ptr<VData>(new VData());
				voxel->getIndexVector()->push_back(scanCounter);
				leafVoxelMap.insert(std::pair<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> (leaf, voxel));
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

//		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;

		// compute Centroid
		typename VData::ScanIndexVectorPtr scanIndexVector = voxel->getIndexVector();
		typename VData::ScanIndexVector::iterator indexItr;

		PointT centroid;
		double x(0), y(0), z(0), r(0), g(0), b(0);
		for (indexItr = scanIndexVector->begin(); indexItr != scanIndexVector->end(); ++indexItr) {
			PointT p = A->at(*indexItr);
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
	cout << "Total leaves with supervoxels " << leafMapping.size() << endl;

	int leavesNotFoundWithSupervoxel = 0;
	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;
		typename SupervoxelClusteringT::LeafContainerT::const_iterator leafItr;

		// check if leaf exists in the mapping from leaf to label
		if (leafMapping.find(leaf) != leafMapping.end()) {

			unsigned int label = leafMapping[leaf];

			// calculate normal for this leaf
			Eigen::Vector4f params = Eigen::Vector4f::Zero();
			float curvature;
			std::vector<int> indicesToConsider;

			for (leafItr = leaf->cbegin(); leafItr != leaf->cend(); ++leafItr) {
				typename SupervoxelClusteringT::LeafContainerT* neighborLeaf = *leafItr;
				if (leafVoxelMap.find(neighborLeaf) != leafVoxelMap.end() &&
						leafMapping.find(neighborLeaf) != leafMapping.end() && leafMapping[neighborLeaf] == label) {	// same supervoxel
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
			if (supervoxelMap.find(label) != supervoxelMap.end()) {
				supervoxelMap[label]->getVoxelAVector()->push_back(voxel);
			} else {

				SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
				supervoxel->setLabel(label);
				supervoxel->getVoxelAVector()->push_back(voxel);

				// Add SV to SVMapping
				supervoxelMap.insert(std::pair<uint, typename SData::Ptr>(label, supervoxel));
			}

			// end if

		} else {
			leavesNotFoundWithSupervoxel++;
		}

	}

	cout << "scan 1 leaves without supervoxel: " << leavesNotFoundWithSupervoxel << endl;

	leafVoxelMap.clear();

	cout<<"Finding supervoxel normals " << endl;
	// calculating supervoxel normal, centroid and covariance
	SVMap::iterator svItr;
	std::vector<int> labelsToRemove;
	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		int supervoxelPointCount = 0;
		int svLabel = svItr->first;
		SData::Ptr supervoxel = svItr->second;
		SData::VoxelVectorPtr voxels = supervoxel->getVoxelAVector();

		Eigen::Vector3f supervoxelNormal = Eigen::Vector3f::Zero();
        Eigen::Vector4f supervoxelCentroid = Eigen::Vector4f::Identity();
        Eigen::Matrix3f supervoxelCovariance = Eigen::Matrix3f::Zero();
        
        VData::ScanIndexVector supervoxelScanIndices; // needed for supervoxel centroid

		typename SData::VoxelVector::iterator voxelItr;
		for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {
            VData::ScanIndexVectorPtr voxelIndices = (*voxelItr)->getIndexVector();
            supervoxelScanIndices.insert(supervoxelScanIndices.begin(), voxelIndices->begin(), voxelIndices->end());
            
			int voxelSize = (*voxelItr)->getIndexVector()->size();
			supervoxelPointCount += voxelSize;
			supervoxelNormal += (*voxelItr)->getNormal();
		}

		if (supervoxelPointCount >= MIN_POINTS_IN_SUPERVOXEL) {

			pcl::compute3DCentroid(*A, supervoxelScanIndices, supervoxelCentroid);
			pcl::computeCovarianceMatrix(*A, supervoxelScanIndices, supervoxelCentroid, supervoxelCovariance);

			Eigen::JacobiSVD<Eigen::Matrix3f> svd (supervoxelCovariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

			Eigen::VectorXf singularValues = svd.singularValues();
			Eigen::MatrixXf U = svd.matrixU();
			Eigen::MatrixXf V = svd.matrixV();

			if (singularValues(1) < singularValues(0) / 10) {
				singularValues(1) = singularValues(0) / 10;
				singularValues(2) = singularValues(0) / 10;
			} else if (singularValues(2) < singularValues(0) / 10) {
				singularValues(2) = singularValues(0) / 10;
			}

			Eigen::MatrixXf S = singularValues.asDiagonal();
			supervoxelCovariance = U * singularValues.asDiagonal() * V.transpose();

			supervoxel->setCovariance(supervoxelCovariance);
			supervoxel->setCentroid(supervoxelCentroid);
			supervoxel->setPointACount(supervoxelPointCount);

			if (!supervoxelNormal.isZero()) {
				supervoxelNormal.normalize();
				supervoxel->setNormal(supervoxelNormal);
			}

		} else {
			labelsToRemove.push_back(svLabel);
		}

	}
    
	cout << "Debug Mode: Removing supervoxels with points less than " << MIN_POINTS_IN_SUPERVOXEL << endl;
	// Remove labels in labelsToRemove
	std::vector<int>::iterator itr = labelsToRemove.begin();
	for (; itr != labelsToRemove.end(); itr ++) {
		supervoxelMap.erase(*itr);
	}


    // Covariance and mean calculated
    
    
	cout << "Calculating supervoxel data - covariance, mean, epsilon1 and epsilon2" << endl;
    // Calculate Epsilon1 and Epsilon2 for each supervoxel where
    // TotalSupervoxelProbability = Epsilon1 * ProbabilityFromNaturalDistribution + Epsilon2 * ProbabilityFromOutilers
    // Epsilon1 + Epsilon2 = 1
    // TotalSupervoxelProbability has mass 1
    
    for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

    	int svLabel = svItr->first;
        SData::Ptr supervoxel = svItr->second;
        SData::VoxelVectorPtr voxels = supervoxel->getVoxelAVector();

        Eigen::Vector4f supervoxelCentroid = supervoxel->getCentroid();
        Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();

        double outlierPro = PROBABILITY_OUTLIERS_SUPERVOXEL;
        double probabilityOutliers =  voxels->size() * svr_util::cube<float>(vr) * outlierPro; // Epsilon2
        double totalProbabilityFromND = 0.0f; // Epsilon1
        double epsilon1(0), epsilon2(0);

        typename SData::VoxelVector::iterator voxelItr;
        for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {

        	VData::Ptr voxel = (*voxelItr);
            float ax, bx, ay, by, az, bz;

            adjTree->getLeafBounds(voxel->getCentroid(), ax, bx, ay, by, az, bz); // using centroid, any point should work
            totalProbabilityFromND += svr_util::calculateApproximateIntegralForVoxel(ax, bx, ay, by, az, bz, supervoxelCovariance, supervoxelCentroid);
        }

        totalProbabilityFromND -= probabilityOutliers;
        epsilon1 = (1-probabilityOutliers) / totalProbabilityFromND;
        epsilon2 = 1-epsilon1;

        supervoxel->setEpsilon1(epsilon1);
        supervoxel->setEpsilon2(epsilon2);

        if (svLabel == 515 || svLabel == 487 || svLabel == 536) {

        	cout << "Label: " << svLabel << endl;
        	cout << "Covariance: " << endl << supervoxelCovariance << endl;
        	cout << "Mean: " << endl << supervoxelCentroid << endl;

        }

        if (epsilon1 > 1 || epsilon2 > 1) {
        	cout << "Supervoxel Label: " << svLabel << " Points: " << supervoxel->getPointACount() <<  " Voxels: " << voxels->size() << endl;
        	cout << "Epsilon 1: " << epsilon1 << endl;
        	cout << "Epsilon 2: " << epsilon2 << endl;
        }

    }

    cout << "Precomputing supervoxel function constants d1, d2, d3 and convariance inverse " << endl;
    // No need seperate loop for this

    for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

    	int svLabel = svItr->first;
    	SData::Ptr supervoxel = svItr->second;

    	Eigen::Vector4f supervoxelCentroid = supervoxel->getCentroid();
    	Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();

    	double normal3DConstant = NORMAL_3D_CONSTANT;

    	double epsilon1 = supervoxel->getEpsilon1();
    	double epsilon2 = supervoxel->getEpsilon2();

    	double c2 = epsilon2 * PROBABILITY_OUTLIERS_SUPERVOXEL;
    	double c1 = epsilon1 * normal3DConstant / sqrt (supervoxelCovariance.determinant()) ;

    	double d3 = -log(c2);
    	double d1 = -log(c1 + c2) -d3;
    	double d2 = -2 * log( (-log(c1 * NSQRT_EXP + c2) - d3) / d1);

    	Eigen::Matrix3f supervoxelCovarianceInverse = supervoxelCovariance.inverse();

    	supervoxel->setD1(d1);
    	supervoxel->setD2(d2);
    	supervoxel->setD3(d3);
    	supervoxel->setCovarianceInverse(supervoxelCovarianceInverse);

    }


}

void
SupervoxelRegistration::createSuperVoxelMappingForScan2 (PointCloudT::Ptr transformedB) {

	AdjacencyOctreeT adjTree1 = supervoxelClustering.getOctreeeAdjacency();

	AdjacencyOctreeT adjTree2;
	adjTree2.reset (new typename SupervoxelClusteringT::OctreeAdjacencyT(vr));
	adjTree2->setInputCloud(transformedB);
	adjTree2->customBoundingBox(octree_bounds_.minPt.x, octree_bounds_.minPt.y, octree_bounds_.minPt.z,
			octree_bounds_.maxPt.x, octree_bounds_.maxPt.y, octree_bounds_.maxPt.z);
	adjTree2->addPointsFromInputCloud();

	cout << "Adjacency Octree Created for scan2 after transformation" << endl;

	SVMap::iterator svItr;

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {
		typename SData::Ptr supervoxel = svItr->second;
		supervoxel->clearScanBMapping();
		supervoxel->clearScanBData();
	}

	pcl::PointCloud<PointT>::iterator scanItr;
	int scanCounter = 0;

	LeafVoxelMapT leafVoxelMap;

	for (scanItr = transformedB->begin(); scanItr != transformedB->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree2 -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree2 -> getLeafContainerAtPoint(a);

			if (leafVoxelMap.find(leaf) != leafVoxelMap.end()) {
				leafVoxelMap[leaf]->getIndexVector()->push_back(scanCounter);
			} else {
				VData::Ptr voxel = boost::shared_ptr<VData>(new VData());
				voxel->getIndexVector()->push_back(scanCounter);
				leafVoxelMap.insert(std::pair<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> (leaf, voxel));
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
			PointT p = transformedB->at(*indexItr);
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
	pcl::PointXYZ queryPoint;
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

			if (leafMapping.find(leaf1) != leafMapping.end()) {

				unsigned int label = leafMapping[leaf1];

				// check if SVMapping already contains the supervoxel
				if (supervoxelMap.find(label) != supervoxelMap.end()) {
					SData::Ptr supervoxel = supervoxelMap[label];
					supervoxel->getVoxelBVector()->push_back(voxel);
				} else {
//					SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
//					supervoxel->setLabel(label);
//					supervoxel->getVoxelBVector()->push_back(voxel);
//					// Add SV to SVMapping
//					supervoxelMap.insert(std::pair<uint, typename SData::Ptr>(label, supervoxel));
				}

			} else {
				// leaf exists in scan 1 but doesn't have a supervoxel label
				searchForSupervoxel = true;
			}

		} else {
			// search for neareset supervoxel for this leaf
			searchForSupervoxel = true;
		}

		searchForSupervoxel = true;
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

			if (supervoxelKdTree.nearestKSearch (queryPoint,NN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
			{

				SData::Ptr supervoxel;
				Eigen::Vector3f supervoxelNormal;

				int closestSupervoxelLabel = -1;
				double minDistance = INT_MAX;
				for (intItr = pointIdxNKNSearch.begin(); intItr != pointIdxNKNSearch.end(); ++intItr) {

					int index = *intItr;

					float euclideanD = sqrt(pointNKNSquaredDistance[index]);
					int svLabel = kdTreeLabels[index];
					supervoxel = supervoxelMap[svLabel];
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
					supervoxelMap[closestSupervoxelLabel]->getVoxelBVector()->push_back(voxel);

			}

			// search for matching normal among the NN supervoxels using new Distance Function:
			// D = (1 - log2( 1 -  acos(SupervoxelNormal dot VoxelNormal) / (Pi/2) )) * Euclidean_Distance
		}
	}

	// end supervoxel iteration
	leafVoxelMap.clear();

	if (debug) {
		calculateSupervoxelScanBData();
	}

}

void
SupervoxelRegistration::calculateSupervoxelScanBData() {

	typename SVMap::iterator svItr;
	typename SData::VoxelVector::iterator voxelItr;
	typename VData::ScanIndexVector::iterator indexItr;

	// calculate centroids for scan B and count of B points
	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		int supervoxelPointCount = 0;
		SData::Ptr supervoxel = svItr->second;
		SData::VoxelVectorPtr voxels = supervoxel->getVoxelBVector();

		for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {
			VData::Ptr voxel = *voxelItr;
			typename VData::ScanIndexVectorPtr indexVector = voxel->getIndexVector();
			int numberOfPoints = indexVector->size();
			supervoxelPointCount += numberOfPoints;
		}

		supervoxel->setPointBCount(supervoxelPointCount);
	}
}

void
SupervoxelRegistration::createKDTreeForSupervoxels() {

	PointCloudXYZ::Ptr svLabelCloud = boost::shared_ptr<PointCloudXYZ>(new PointCloudXYZ());

	SVMap::iterator svItr;
	
	pcl::PointXYZ treePoint;

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		SData::Ptr supervoxel = svItr->second;
		int label = svItr->first;

        Eigen::Vector4f centroid = supervoxel->getCentroid();

        PointT supervoxelCentroid;
        supervoxelCentroid.x = centroid[0];
        supervoxelCentroid.y = centroid[1];
        supervoxelCentroid.z = centroid[2];

		treePoint.x = supervoxelCentroid.x;
		treePoint.y = supervoxelCentroid.y;
		treePoint.z = supervoxelCentroid.z;
		kdTreeLabels.push_back(label);

		svLabelCloud->push_back(treePoint);
	}

	supervoxelKdTree.setInputCloud(svLabelCloud);
}

void
SupervoxelRegistration::showPointClouds(std::string viewerTitle) {

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer (viewerTitle));
	viewer->setBackgroundColor (0,0,0);

	std::string id1("scan1"), id2("scan2");

	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(A);
	viewer->addPointCloud<PointT> (A, rgb1, id1);

	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(B);
	viewer->addPointCloud<PointT> (B, rgb2, id2);

	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id2);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}

void
SupervoxelRegistration::showPointCloud(typename PointCloudT::Ptr scan) {

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Supervoxel Based MI Viewer"));
	viewer->setBackgroundColor (0,0,0);

	std::string id1("scan");

	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(scan);
	viewer->addPointCloud<PointT> (scan, rgb1, id1);

	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}

void
SupervoxelRegistration::showTestSuperVoxel(int supevoxelLabel) {

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxelClusters = initializeVoxels();
	std::cout << "Number of supervoxels: " << supervoxelMap.size() << std::endl;

	createSuperVoxelMappingForScan1();
	createKDTreeForSupervoxels();
	createSuperVoxelMappingForScan2(B);

	// Display all supervoxels with count A and count B

	SVMap::iterator svItr;
	SData::VoxelVector::iterator voxelItr;
	VData::ScanIndexVector::iterator indexItr;

	int SV = supevoxelLabel;
	typename PointCloudT::Ptr newCloud (new PointCloudT);

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

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

					PointT p = A->at(*indexItr);

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

					PointT p = B->at(*indexItr);

					p.r = 0;
					p.g = 255;
					p.b = 0;

					newCloud->push_back(p);
				}


			}
		}

//		cout << svLabel << '\t' << "A: " << counterA << '\t' << "B: " << counterB << endl;
	}

	showPointCloud(newCloud);
}







