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
#include <pcl/visualization/point_picking_event.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>

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
	appx = false;
	debug = false;
	octree_bounds_.minPt.x = -120;
	octree_bounds_.minPt.y = -120;
	octree_bounds_.minPt.z = -100;

	octree_bounds_.maxPt.x = 120;
	octree_bounds_.maxPt.y = 120;
	octree_bounds_.maxPt.z = 100;
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
	createSuperVoxelMappingForScan1();
	createKDTreeForSupervoxels();
	calculateScan2Normals();
	if (_SVR_DEBUG_)
		std::cout << "Number of supervoxels: " << supervoxelMap.size() << std::endl;
}

void
SupervoxelRegistration::calculateScan2Normals() {

	if (_SVR_DEBUG_)
		cout << "Computing Scan 2 Normals" << endl;

	// These can be calculated on as per need basis
	// GPU implementation will anyway be ultra fast

	pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
	ne.setInputCloud(B);

	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
	ne.setSearchMethod(tree);

	PointCloudNormal::Ptr tempNormals (new PointCloudNormal());
	normalsB.reset(new PointCloudXYZ ());
	double radius = NORMAL_RADIUS;
	ne.setRadiusSearch(radius); // meters

	ne.compute(*tempNormals);

	PointCloudNormal::iterator pItr;
	for (pItr = tempNormals->begin(); pItr != tempNormals->end(); ++pItr) {
		pcl::PointXYZ p;
		p.x = pItr->normal_x;
		p.y = pItr->normal_y;
		p.z = pItr->normal_z;
		normalsB->push_back(p);
	}

	if (_SVR_DEBUG_) {
		cout << "Normals DS Size: " << normalsB->size() << endl;
	}
}

// Return trans
void
SupervoxelRegistration::alignScans(Eigen::Affine3d& final_transform, Eigen::Affine3d& initial_transform) {

	prepareForRegistration();

	if (svr::_SVR_DEBUG_) {
		if (appx)
			cout << "Approximation Enabled" << endl;
		else
			cout << "Approximation not enabled" << endl;
		cout << "Debug Mode: Preparation Complete" << endl;
	}

	float lastItrCost = 0;
	Eigen::Affine3d trans_last = initial_transform;
	Eigen::Affine3d trans_new;
	PointCloudT::Ptr transformedScan2 = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());
	bool converged = false;
	int iteration = 0;
	double delta;
	bool debug = false;
	double epsilon = 5e-4;
	double epsilon_rot = 2e-3;
	double epsilon_cost = 2e-1;
	int maxIteration = 20;

	cout << "Initial T: " << std::endl << trans_last.matrix(); cout << std::endl;

	while (!converged) {

		// transform point cloud using trans_last
		transformPointCloud (*B, *transformedScan2, trans_last);

		Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = trans_last.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan2->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan2, transformedNormalsB, iteration+1);

		// New mapping

		if (svr::_SVR_DEBUG_) {
			cout << "Debug Mode: Printing supervoxel Map..." << endl;
			cout << "Transformation: " << std::endl << trans_last.matrix(); cout << std::endl;
			printSupervoxelMap(iteration+1, debugScanString, trans_last, lastItrCost);
			cout << "Iteration " << iteration+1 << " ..." << endl;
		}

		svr_optimize::svr_opti_data opti_data;
		opti_data.scan2 = B;
		opti_data.svMap = &supervoxelMap;
		opti_data.t = trans_last;
		opti_data.approx = appx;
		float cost;

		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.optimizeUsingGaussNewton(trans_new, cost);
		//		optimize(opti_data, trans_new, cost);

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

		//		float costDiff = fabs(cost - lastItrCost) / epsilon_cost ;
		//		if (costDiff > delta)
		//			delta = costDiff;
		if(iteration >= maxIteration || delta < 1) {
			converged = true;
		}

		lastItrCost = cost;
		trans_last = trans_new;
	}

	final_transform = trans_new;
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
	supervoxelClustering.getLabeledLeafContainerMap(_leafToLabelMapping);
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

	if (_SVR_DEBUG_) {
		cout << "Finding mapping " << endl;
		cout << "Total leaves: " << leafVoxelMap.size() << endl;
		cout << "Total leaves with supervoxels " << _leafToLabelMapping.size() << endl;
	}

	int leavesNotFoundWithSupervoxel = 0;
	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;
		typename SupervoxelClusteringT::LeafContainerT::const_iterator leafItr;

		// check if leaf exists in the mapping from leaf to label
		if (_leafToLabelMapping.find(leaf) != _leafToLabelMapping.end()) {

			unsigned int label = _leafToLabelMapping[leaf];

			// calculate normal for this leaf
			Eigen::Vector4f params = Eigen::Vector4f::Zero();
			float curvature;
			std::vector<int> indicesToConsider;

			for (leafItr = leaf->cbegin(); leafItr != leaf->cend(); ++leafItr) {

				// first neighbour
				typename SupervoxelClusteringT::LeafContainerT* neighborLeaf = *leafItr;
				if (leafVoxelMap.find(neighborLeaf) != leafVoxelMap.end() &&
						_leafToLabelMapping.find(neighborLeaf) != _leafToLabelMapping.end() && _leafToLabelMapping[neighborLeaf] == label) {	// same supervoxel
					VData::Ptr neighborVoxel = leafVoxelMap[neighborLeaf];
					indicesToConsider.push_back(neighborVoxel->getCentroidCloudIndex());
				}

				// second neighbours
				typename SupervoxelClusteringT::LeafContainerT::const_iterator neighbouLeafItr;
				for (neighbouLeafItr = neighborLeaf->cbegin(); neighbouLeafItr != neighborLeaf->cend(); ++neighbouLeafItr) {
					typename SupervoxelClusteringT::LeafContainerT* secondNeighborLeaf = *neighbouLeafItr;

					if (leafVoxelMap.find(secondNeighborLeaf) != leafVoxelMap.end() &&
							_leafToLabelMapping.find(secondNeighborLeaf) != _leafToLabelMapping.end() && _leafToLabelMapping[secondNeighborLeaf] == label) {	// same supervoxel
						VData::Ptr secondNeighborVoxel = leafVoxelMap[secondNeighborLeaf];
						indicesToConsider.push_back(secondNeighborVoxel->getCentroidCloudIndex());
					}
				}

			}

			pcl::computePointNormal(centroidCloud, indicesToConsider, params, curvature);
			pcl::flipNormalTowardsViewpoint(voxel->getCentroid(), 0, 0, 0, params);

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

	if (_SVR_DEBUG_)
		cout << "scan 1 leaves without supervoxel: " << leavesNotFoundWithSupervoxel << endl;

	leafVoxelMap.clear();

	if (_SVR_DEBUG_)
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
		Eigen::Vector4f supervoxelCentroid = Eigen::Vector4f::Zero();
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

			//			supervoxelNormal = U.col(2);
			pcl::PointXYZ c;
			c.x = supervoxelCentroid(0);
			c.y = supervoxelCentroid(1);
			c.z = supervoxelCentroid(2);
			//			pcl::flipNormalTowardsViewpoint(c, 0, 0, 0, supervoxelNormal);


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

	if (_SVR_DEBUG_)
		cout << "Debug Mode: Removing " <<  labelsToRemove.size() << " supervoxels with points less than " << MIN_POINTS_IN_SUPERVOXEL << endl;

	// Remove labels in labelsToRemove
	std::vector<int>::iterator itr = labelsToRemove.begin();
	for (; itr != labelsToRemove.end(); itr ++) {
		supervoxelMap.erase(*itr);
	}


	// Covariance and mean calculated

	if (_SVR_DEBUG_)
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
		double probabilityOutliers =  voxels->size() * svr_util::cube<float>(vr) * outlierPro; // Integration over supervoxel, Epsilon2
		double totalProbabilityFromND = 0.0f; // Epsilon1
		double epsilon1(0), epsilon2(0);

		typename SData::VoxelVector::iterator voxelItr;

		for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {

			VData::Ptr voxel = (*voxelItr);
			float ax, bx, ay, by, az, bz;

			adjTree->getLeafBounds(voxel->getCentroid(), ax, bx, ay, by, az, bz); // using centroid, any point should work

			double voxelIntegral = svr_util::calculateApproximateIntegralForVoxel(ax, bx, ay, by, az, bz, supervoxelCovariance, supervoxelCentroid);
			totalProbabilityFromND += voxelIntegral;
		}

		totalProbabilityFromND -= probabilityOutliers;
		epsilon1 = (1-probabilityOutliers) / totalProbabilityFromND;
		epsilon2 = 1-epsilon1;

		supervoxel->setEpsilon1(epsilon1);
		supervoxel->setEpsilon2(epsilon2);

		if (epsilon1 >= 1 || epsilon2 >= 1) {
			cout << "Supervoxel Label: " << svLabel << " Points: " << supervoxel->getPointACount() <<  " Voxels: " << voxels->size() << endl;
			cout << "Epsilon 1: " << epsilon1 << endl;
			cout << "Epsilon 2: " << epsilon2 << endl;
		}

	}

	if (_SVR_DEBUG_)
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

		//    	cout << "Label: " << svLabel << std::endl;
		//    	cout << "d1: " << d1 << std::endl;
		//    	cout << "d2: " << d2 << std::endl;
		//    	cout << "d3: " << d3 << std::endl;

		supervoxel->setD1(d1);
		supervoxel->setD2(d2);
		supervoxel->setD3(d3);
		supervoxel->setCovarianceInverse(supervoxelCovarianceInverse);

	}


}

void
SupervoxelRegistration::createSuperVoxelMappingForScan2 (PointCloudT::Ptr transformedB, PointCloudXYZ::Ptr transformedNormalsB, int iteration) {

	SVMap::iterator svItr;

	cout << "Find correspondences for scan B" << endl;

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {
		typename SData::Ptr supervoxel = svItr->second;
		supervoxel->clearScanBMapping();
		supervoxel->clearScanBData();
	}

	AdjacencyOctreeT adjTree1 = supervoxelClustering.getOctreeeAdjacency();

	// Setup search params
	int NN = SEARCH_SUPERVOXEL_NN;
	pcl::PointXYZ queryPoint;
	std::vector<int> pointIdxNKNSearch(NN);
	std::vector<float> pointNKNSquaredDistance(NN);
	std::vector<int>::iterator intItr;

	for (int i = 0; i < transformedB->size(); i++) {

		PointT p = transformedB->at(i);

		bool searchForSupervoxel = false;
		bool presentInScan1 = adjTree1-> isVoxelOccupiedAtPoint(p);

		if (presentInScan1) {
			// scan1 leaf
			SupervoxelClusteringT::LeafContainerT* leaf1 = adjTree1->getLeafContainerAtPoint(p);

			if (_leafToLabelMapping.find(leaf1) != _leafToLabelMapping.end()) {

				unsigned int label = _leafToLabelMapping[leaf1];
				// check if SVMapping already contains the supervoxel
				if (supervoxelMap.find(label) != supervoxelMap.end()) {
					SData::Ptr supervoxel = supervoxelMap[label];
					supervoxel->getScanBIndexVector()->push_back(i);
				}
				else
					searchForSupervoxel = true;

			} else {
				// leaf exists in scan 1 but doesn't have a supervoxel label
				searchForSupervoxel = true;
			}

		} else {
			// search for neareset supervoxel for this leaf
			searchForSupervoxel = true;
		}

		if (searchForSupervoxel) {

			Eigen::Vector3f normal;
			normal[0] = transformedNormalsB->at(i).x;
			normal[1] = transformedNormalsB->at(i).y;
			normal[2] = transformedNormalsB->at(i).z;

			// Clear prev indices
			pointIdxNKNSearch.clear();
			pointNKNSquaredDistance.clear();

			queryPoint.x = p.x;
			queryPoint.y = p.y;
			queryPoint.z = p.z;

			// search for NN centroids

			if (supervoxelKdTree.nearestKSearch (queryPoint,NN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
			{

				SData::Ptr supervoxel;
				Eigen::Vector3f supervoxelNormal;

				int closestSupervoxelLabel = -1;
				double minDistance = INT_MAX;
//				double minDistance = 50;
				int counter = 0;
				for (intItr = pointIdxNKNSearch.begin(); intItr != pointIdxNKNSearch.end(); ++intItr, ++counter) {

					int supervoxelLabelIndex = *intItr;

					float euclideanD = sqrt(pointNKNSquaredDistance[counter]);
					int svLabel = kdTreeLabels[supervoxelLabelIndex];
					supervoxel = supervoxelMap[svLabel];
					supervoxelNormal = supervoxel->getNormal();

					double d = acos(normal.dot(supervoxelNormal)) * 2 / M_PI;
					d = 1 - log2(1.0 - d);
					d *= euclideanD;

					if (d < minDistance) {
						minDistance = d;
						closestSupervoxelLabel = svLabel;
					}

				}

				if (closestSupervoxelLabel > 0)
					supervoxelMap[closestSupervoxelLabel]->getScanBIndexVector()->push_back(i);

			}

			// search for matching normal among the NN supervoxels using new Distance Function:
			// D = (1 - log2( 1 -  acos(SupervoxelNormal dot VoxelNormal) / (Pi/2) )) * Euclidean_Distance
		}
	}


	if (svr::_SVR_DEBUG_)
		calculateSupervoxelScanBData();
}

void
SupervoxelRegistration::displayPointWithPossibleSupervoxelCorrespondences(PointT query, std::vector<int> pointIdxNKNSearchVector) {

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
		SData::ScanIndexVectorPtr indexVector = supervoxel->getScanBIndexVector();

		int numberOfPoints = indexVector->size();
		supervoxelPointCount += numberOfPoints;
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

void static
pp_callback(const pcl::visualization::PointPickingEvent& event, void* viewer)
{
	std::cout << "Picking event active" << std::endl;
	if(event.getPointIndex()!=-1)
	{
		float x,y,z;
		event.getPoint(x,y,z);
		std::cout << x<< ";" << y<<";" << z << std::endl;
	}
}

void
SupervoxelRegistration::showPointCloud(typename PointCloudT::Ptr scan) {

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Supervoxel Based MI Viewer"));
	viewer->setBackgroundColor (0,0,0);
	viewer->registerPointPickingCallback(pp_callback, (void*) &*scan);
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
SupervoxelRegistration::showNormalPointCloud(PointCloudNormal::Ptr scan, PointCloudT::Ptr scanPoints) {

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Supervoxel Based Viewer For Normals"));
	viewer->setBackgroundColor (0,0,0);
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(scanPoints);
	viewer->addPointCloud<PointT> (scanPoints, rgb, "point cloud");
	viewer->addPointCloudNormals<PointT, Normal> (scanPoints, scan, 10, 0.05, "normal cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}


void
SupervoxelRegistration::showTestSuperVoxel(int supevoxelLabel) {

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxelClusters = initializeVoxels();

	createSuperVoxelMappingForScan1();
	std::cout << "Number of supervoxels: " << supervoxelMap.size() << std::endl;

	createKDTreeForSupervoxels();
	calculateScan2Normals();
	createSuperVoxelMappingForScan2(B, normalsB, 1);

	// Display all supervoxels with count A and count B

	SVMap::iterator svItr;
	SData::VoxelVector::iterator voxelItr;
	VData::ScanIndexVector::iterator indexItr;

	int SV = supevoxelLabel;
	typename PointCloudT::Ptr newCloud (new PointCloudT);
	typename PointCloudT::Ptr meanCloud (new PointCloudT);

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		int svLabel = svItr->first;
		typename SData::Ptr supervoxel = svItr->second;
		bool showSupervoxel = false;

		if (svLabel == SV)
			showSupervoxel = true;

		SData::VoxelVectorPtr voxelsA = supervoxel->getVoxelAVector();

		int counterA(0), counterB(0);

		int colorR = rand() / 255;
		int colorG = rand() / 255;
		int colorB = rand() / 255;

		Eigen::Vector4f meanVector = supervoxel->getCentroid();
		PointT meanP;
		meanP.x = meanVector(0);
		meanP.y = meanVector(1);
		meanP.z = meanVector(2);
		meanP.r = colorG;
		meanP.g = colorR;
		meanP.b = colorB;

		meanCloud->push_back(meanP);


		for (voxelItr = voxelsA->begin(); voxelItr != voxelsA->end(); ++ voxelItr) {
			VData::Ptr voxel = (*voxelItr);
			counterA+= voxel->getIndexVector()->size();

			//			if (showSupervoxel) {
			//
			//				for (indexItr = voxel->getIndexVector()->begin(); indexItr != voxel->getIndexVector()->end(); ++indexItr) {
			//
			//					PointT p = A->at(*indexItr);
			//
			//					p.r = 255;
			//					p.g = 0;
			//					p.b = 0;
			//
			//					newCloud->push_back(p);
			//				}
			//
			//
			//			}
			//			else {
			//
			for (indexItr = voxel->getIndexVector()->begin(); indexItr != voxel->getIndexVector()->end(); ++indexItr) {

				PointT p = A->at(*indexItr);

				p.r = colorG;
				p.g = colorR;
				p.b = colorB;

				newCloud->push_back(p);
			}

		}

		for (indexItr = supervoxel->getScanBIndexVector()->begin(); indexItr != supervoxel->getScanBIndexVector()->end(); ++indexItr) {

			PointT p = B->at(*indexItr);

			p.r = colorG;
			p.g = colorR;
			p.b = colorB;

			newCloud->push_back(p);
		}

		//		if (showSupervoxel) {
		//
		//			for (indexItr = supervoxel->getScanBIndexVector()->begin(); indexItr != supervoxel->getScanBIndexVector()->end(); ++indexItr) {
		//
		//				PointT p = B->at(*indexItr);
		//
		//				p.r = 0;
		//				p.g = 255;
		//				p.b = 0;
		//
		//				newCloud->push_back(p);
		//			}
		//		}

		counterB += supervoxel->getScanBIndexVector()->size();
		cout << svLabel << '\t' << "A: " << counterA << '\t' << "B: " << counterB << endl;
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Supervoxel Based MI Viewer"));
	viewer->setBackgroundColor (0,0,0);
	viewer->registerPointPickingCallback(pp_callback, (void*) &*newCloud);
	std::string id1("scan");
	std::string id2("meanScan");

	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(newCloud);
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(meanCloud);
	viewer->addPointCloud<PointT> (newCloud, rgb1, id1);
	viewer->addPointCloud<PointT> (meanCloud, rgb2, id2);

	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, id2);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

//	showPointCloud(newCloud);
}

void SupervoxelRegistration::preparePlotData() {

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxelClusters = initializeVoxels();
	createSuperVoxelMappingForScan1();
	std::cout << "Number of supervoxels: " << supervoxelMap.size() << std::endl;

	createKDTreeForSupervoxels();
	calculateScan2Normals();

}

void SupervoxelRegistration::plotCostFunctionForX(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double actualX;
	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform, normalTransformation;
	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	actualX = x;
	x = actualX - range;

	while (x <= (actualX+range)) {

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		// Normal Transformation
		normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = transform.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

		svr_optimize::svr_opti_data opti_data;
		opti_data.scan2 = B;
		opti_data.svMap = &supervoxelMap;
		double cost;

		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(x, cost));
		std::cout << "x: " << x << " Cost: " << cost << std::endl;

		x += step;
	}

	std::string XString = (boost::format("X:%f")%actualX).str();
	plotter->addPlotData(plotData, XString.c_str());

}

void SupervoxelRegistration::plotCostFunctionForY(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform, normalTransformation;
	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	double actualY = y;
	y = actualY - range;

	while (y <= (actualY + range)) {

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		// Normal Transformation
		normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = transform.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

		svr_optimize::svr_opti_data opti_data;
		opti_data.scan2 = B;
		opti_data.svMap = &supervoxelMap;
		double cost;

		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(y, cost));
		std::cout << "y: " << y << " Cost: " << cost << std::endl;

		y += step;
	}

	std::string ystring = (boost::format("Y:%f")%actualY).str();
	plotter->addPlotData(plotData, ystring.c_str());

}

void SupervoxelRegistration::plotCostFunctionForZ(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform, normalTransformation;
	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	double actualZ = z;
	z = actualZ - range;

	while (z <= (actualZ + range)) {

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		// Normal Transformation
		normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = transform.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

		svr_optimize::svr_opti_data opti_data;
		opti_data.scan2 = B;
		opti_data.svMap = &supervoxelMap;
		double cost;

		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(z, cost));
		std::cout << "z: " << z << " Cost: " << cost << std::endl;

		z += step;
	}

	std::string zstring = (boost::format("Z:%f")%actualZ).str();
	plotter->addPlotData(plotData, zstring.c_str());

}

void SupervoxelRegistration::plotCostFunctionForRoll(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& rangeDegree,
		float& stepDegree) {

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform, normalTransformation;
	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	double actualRollDeg = roll * 180 / M_PI;
	double rollDegree = actualRollDeg - rangeDegree;

	while (rollDegree <= (actualRollDeg + rangeDegree)) {

		double rollVar = rollDegree * M_PI / 180;

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (rollVar, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		// Normal Transformation
		normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = transform.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

		svr_optimize::svr_opti_data opti_data;
		opti_data.svMap = &supervoxelMap;
		double cost = 0;
		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(rollDegree, cost));
		std::cout << "roll in degrees: " << rollDegree << " Cost: " << cost << std::endl;

		rollDegree += stepDegree;
	}

	std::string rollString = (boost::format("ROLL:%f")%actualRollDeg).str();
	plotter->addPlotData(plotData, rollString.c_str());

}

void SupervoxelRegistration::plotCostFunctionForPitch(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& rangeDegree,
		float& stepDegree) {

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform, normalTransformation;
	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	double actualPitchDeg = pitch * 180 / M_PI;
	double pitchDegree = actualPitchDeg - rangeDegree;

	while (pitchDegree <= (actualPitchDeg + rangeDegree)) {

		double pitchVar = pitchDegree * M_PI / 180;

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitchVar, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		// Normal Transformation
		normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = transform.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

		svr_optimize::svr_opti_data opti_data;
		opti_data.svMap = &supervoxelMap;
		double cost = 0;
		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(pitchDegree, cost));
		std::cout << "pitch in Degrees: " << pitchDegree << " Cost: " << cost << std::endl;

		pitchDegree += stepDegree;
	}

	std::string pitchString = (boost::format("PITCH:%f")%actualPitchDeg).str();
	plotter->addPlotData(plotData, pitchString.c_str());

}

void SupervoxelRegistration::plotCostFunctionForYaw(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& rangeDegree,
		float& stepDegree) {

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform, normalTransformation;
	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	double actualYawDeg = yaw * 180 / M_PI;
	double yawDegree = actualYawDeg - rangeDegree;

	while (yawDegree <= (actualYawDeg + rangeDegree)) {

		double yawVar = yawDegree * M_PI / 180;

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yawVar, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		// Normal Transformation
		normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = transform.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

		svr_optimize::svr_opti_data opti_data;
		opti_data.svMap = &supervoxelMap;
		double cost = 0;
		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(yawDegree, cost));
		std::cout << "yaw in degrees: " << yawDegree << " Cost: " << cost << std::endl;

		yawDegree += stepDegree;
	}

	std::string yawstring = (boost::format("YAW:%f")%actualYawDeg).str();
	plotter->addPlotData(plotData, yawstring.c_str());

}

void SupervoxelRegistration::plotOptimizationIterationsForX(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	transformPointCloud (*B, *transformedScan, baseT);
	Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

	// copy rotation matrix
	for(int k = 0; k < 3; k++) {
		for(int l = 0; l < 3; l++) {
			normalTransformation.matrix()(k,l) = baseT.matrix()(k,l);
		}
	}

	transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);
	for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
		pcl::PointXYZ pN = transformedNormalsB->at(pi);
		pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
	}

	createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);



	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);
	std::cout << "Initial x:" << x << std::endl;
	std::cout << "Initial y:" << y << std::endl;
	std::cout << "Initial z:" << z << std::endl;
	std::cout << "Initial roll:" << roll << std::endl;
	std::cout << "Initial pitch:" << pitch << std::endl;
	std::cout << "Initial yaw:" << yaw << std::endl;

	Eigen::Affine3d transform;

	double baseX = x;
	x -= range;

	svr_optimize::svr_opti_data opti_data;
	opti_data.svMap = &supervoxelMap;

	svr_optimize::svrOptimize optimizer;
	optimizer.setOptimizeData(opti_data);
	while (x <= (baseX+range)) {

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		double cost = 0;
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(x, cost));
		std::cout << "x: " << x << " Cost: " << cost << std::endl;

		x += step;
	}

	plotter->addPlotData(plotData, "WRT_X");

}

void SupervoxelRegistration::plotOptimizationIterationsForY(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	transformPointCloud (*B, *transformedScan, baseT);
	Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

	// copy rotation matrix
	for(int k = 0; k < 3; k++) {
		for(int l = 0; l < 3; l++) {
			normalTransformation.matrix()(k,l) = baseT.matrix()(k,l);
		}
	}

	transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);
	for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
		pcl::PointXYZ pN = transformedNormalsB->at(pi);
		pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
	}

	createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform;

	double baseY = y;
	y = baseY - range;

	svr_optimize::svr_opti_data opti_data;
	opti_data.svMap = &supervoxelMap;

	svr_optimize::svrOptimize optimizer;
	optimizer.setOptimizeData(opti_data);

	while (y <= (baseY+range)) {

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		double cost = 0;
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(y, cost));
		std::cout << "y: " << y << " Cost: " << cost << std::endl;

		y += step;
	}

	plotter->addPlotData(plotData, "WRT_Y");

}

void SupervoxelRegistration::plotOptimizationIterationsForZ(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	transformPointCloud (*B, *transformedScan, baseT);
	Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

	// copy rotation matrix
	for(int k = 0; k < 3; k++) {
		for(int l = 0; l < 3; l++) {
			normalTransformation.matrix()(k,l) = baseT.matrix()(k,l);
		}
	}

	transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);
	for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
		pcl::PointXYZ pN = transformedNormalsB->at(pi);
		pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
	}

	createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform;

	double baseZ = z;

	z = baseZ - range;
	svr_optimize::svr_opti_data opti_data;
	opti_data.svMap = &supervoxelMap;

	svr_optimize::svrOptimize optimizer;
	optimizer.setOptimizeData(opti_data);

	while (z <= (baseZ + range)) {

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		double cost = 0;
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(z, cost));
		std::cout << "z: " << z << " Cost: " << cost << std::endl;

		z += step;
	}

	plotter->addPlotData(plotData, "WRT_Z");

}

void SupervoxelRegistration::plotOptimizationIterationsForRoll(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	transformPointCloud (*B, *transformedScan, baseT);
	Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

	// copy rotation matrix
	for(int k = 0; k < 3; k++) {
		for(int l = 0; l < 3; l++) {
			normalTransformation.matrix()(k,l) = baseT.matrix()(k,l);
		}
	}

	transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);
	for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
		pcl::PointXYZ pN = transformedNormalsB->at(pi);
		pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
	}

	createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform;

	double baseRollDeg = roll * 180 / M_PI;
	double rollDegree = baseRollDeg - range;
	svr_optimize::svr_opti_data opti_data;
	opti_data.svMap = &supervoxelMap;

	svr_optimize::svrOptimize optimizer;
	optimizer.setOptimizeData(opti_data);

	while (rollDegree <= (baseRollDeg + range)) {

		double rollVar = rollDegree * M_PI / 180;

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (rollVar, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		double cost = 0;
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(rollDegree, cost));
		std::cout << "roll: " << rollDegree << " Cost: " << cost << std::endl;

		rollDegree += step;
	}

	plotter->addPlotData(plotData, "WRT_ROLL");

}

void SupervoxelRegistration::plotOptimizationIterationsForPitch(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

	transformPointCloud (*B, *transformedScan, baseT);
	Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

	// copy rotation matrix
	for(int k = 0; k < 3; k++) {
		for(int l = 0; l < 3; l++) {
			normalTransformation.matrix()(k,l) = baseT.matrix()(k,l);
		}
	}

	transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);
	for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
		pcl::PointXYZ pN = transformedNormalsB->at(pi);
		pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
	}

	createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform;

	double basePitchDegree = pitch * 180 / M_PI;
	double pitchDeg = basePitchDegree - range;

	svr_optimize::svr_opti_data opti_data;
	opti_data.svMap = &supervoxelMap;

	svr_optimize::svrOptimize optimizer;
	optimizer.setOptimizeData(opti_data);

	while (pitchDeg <= (basePitchDegree + range)) {

		double pitchVar = pitchDeg * M_PI / 180;

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitchVar, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		double cost = 0;
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(pitchDeg, cost));
		std::cout << "pitch: " << pitchDeg << " Cost: " << cost << std::endl;

		pitchDeg += step;
	}

	plotter->addPlotData(plotData, "WRT_PITCH");

}

void SupervoxelRegistration::plotOptimizationIterationsForYaw(
		pcl::visualization::PCLPlotter* plotter,
		Eigen::Affine3d& baseT,
		float& range,
		float& step) {

	PointCloudT::Ptr transformedScan = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());

//	transformPointCloud (*B, *transformedScan, baseT);
//	Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();
//
//	// copy rotation matrix
//	for(int k = 0; k < 3; k++) {
//		for(int l = 0; l < 3; l++) {
//			normalTransformation.matrix()(k,l) = baseT.matrix()(k,l);
//		}
//	}
//
//	transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);
//	for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
//		pcl::PointXYZ pN = transformedNormalsB->at(pi);
//		pcl::flipNormalTowardsViewpoint(transformedScan->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
//	}

//	createSuperVoxelMappingForScan2(transformedScan, transformedNormalsB, 1);

	createSuperVoxelMappingForScan2(B, normalsB, 1);

	std::vector<std::pair<double, double> > plotData;

	// do for plotStart to plotEnd at ever plotStep increment

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(baseT, &x, &y, &z);
	svr_util::transform_get_rotation_xyz_from_affine(baseT, &roll, &pitch, &yaw);

	Eigen::Affine3d transform;

	double baseYawDeg = yaw * 180 / M_PI;
	double yawDeg = baseYawDeg - range;

	svr_optimize::svr_opti_data opti_data;
	opti_data.svMap = &supervoxelMap;

	svr_optimize::svrOptimize optimizer;
	optimizer.setOptimizeData(opti_data);

	while (yawDeg <= (baseYawDeg + range)) {

		double yawVar = yawDeg * M_PI / 180;

		// Point Cloud transformation
		transform = Eigen::Affine3d::Identity();
		transform.translation() << x,y,z;
		transform.rotate(Eigen::AngleAxisd (yawVar, Eigen::Vector3d::UnitZ()));
		transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));

		transformPointCloud (*B, *transformedScan, transform);

		double cost = 0;
		optimizer.computeCost(cost, transformedScan);

		plotData.push_back(std::pair<double, double>(yawDeg, cost));
		std::cout << "yaw: " << yawDeg << " Cost: " << cost << std::endl;

		yawDeg += step;
	}

	plotter->addPlotData(plotData, "WRT_YAW");

}


// Return trans
void
SupervoxelRegistration::alignScansOriginal(Eigen::Affine3d& final_transform, Eigen::Affine3d& initial_transform) {

	prepareForRegistration();

	float lastItrCost = 0;
	Eigen::Affine3d trans_last = initial_transform;
	Eigen::Affine3d trans_new;
	PointCloudT::Ptr transformedScan2 = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudXYZ::Ptr transformedNormalsB = boost::shared_ptr <PointCloudXYZ> (new PointCloudXYZ ());
	bool converged = false;
	int iteration = 0;
	double delta;
	bool debug = false;
	double epsilon = 5e-4;
	double epsilon_rot = 2e-3;
	double epsilon_cost = 2e-1;
	int maxIteration = 30;

	cout << "Initial T: " << std::endl << trans_last.matrix(); cout << std::endl;

	while (!converged) {

		// transform point cloud using trans_last
		transformPointCloud (*B, *transformedScan2, trans_last);

		Eigen::Affine3d normalTransformation = Eigen::Affine3d::Identity();

		// copy rotation matrix
		for(int k = 0; k < 3; k++) {
			for(int l = 0; l < 3; l++) {
				normalTransformation.matrix()(k,l) = trans_last.matrix()(k,l);
			}
		}

		transformPointCloud (*normalsB, *transformedNormalsB, normalTransformation);

		for (int pi = 0; pi<transformedNormalsB->size(); ++pi) {
			pcl::PointXYZ pN = transformedNormalsB->at(pi);
			pcl::flipNormalTowardsViewpoint(transformedScan2->at(pi), 0, 0, 0, pN.x, pN.y, pN.z);
		}

		createSuperVoxelMappingForScan2(transformedScan2, transformedNormalsB, iteration+1);

		// New mapping

		if (svr::_SVR_DEBUG_) {
//			cout << "Debug Mode: Printing supervoxel Map..." << endl;
//			cout << "Transformation: " << std::endl << trans_last.matrix(); cout << std::endl;
			printSupervoxelMap(iteration+1, debugScanString, trans_last, lastItrCost);
			cout << "Iteration " << iteration+1 << " ..." << endl;
		}

		svr_optimize::svr_opti_data opti_data;
		opti_data.scan2 = B;
		opti_data.svMap = &supervoxelMap;
		opti_data.t = trans_last;
		opti_data.approx = appx;
		float cost;

		svr_optimize::svrOptimize optimizer;
		optimizer.setOptimizeData(opti_data);
		optimizer.optimizeUsingOriginalLMA(trans_new, cost);

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

		//		float costDiff = fabs(cost - lastItrCost) / epsilon_cost ;
		//		if (costDiff > delta)
		//			delta = costDiff;
		if(iteration >= maxIteration || delta < 1) {
			converged = true;
		}

		lastItrCost = cost;
		trans_last = trans_new;
	}

	final_transform = trans_new;
}

