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
#include <gsl/gsl_multimin.h>

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
#include "supervoxel_util.hpp"


using namespace svr;

SupervoxelRegistration::SupervoxelRegistration(float voxelR, float seedR):
	supervoxelClustering (voxelR, seedR),
	vr (voxelR),
	sr (seedR),
	octree_bounds_()
{

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
void
SupervoxelRegistration::alignScans() {

	prepareForRegistration();

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
		createSuperVoxelMappingForScan2();

		cout << "Iteration " << iteration+1 << " ..." << endl;
		trans_new = optimize();

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

	std::string transFilename ("trans_result");
	ofstream fout(transFilename.c_str());
	fout << trans_new.inverse().matrix();
	fout.close();

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
	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		int supervoxelPointCount = 0;
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

        pcl::compute3DCentroid(*A, supervoxelScanIndices, supervoxelCentroid);
        pcl::computeCovarianceMatrix(*A, supervoxelScanIndices, supervoxelCentroid, supervoxelCovariance);
        
        supervoxel->setCovariance(supervoxelCovariance);
        supervoxel->setCentroid(supervoxelCentroid);
		supervoxel->setPointACount(supervoxelPointCount);

		if (!supervoxelNormal.isZero()) {
			supervoxelNormal.normalize();
			supervoxel->setNormal(supervoxelNormal);
		}
        
	}
    
    // Covariance and mean calculated
    
    
    // Calculate Epsilon1 and Epsilon2 for each supervoxel where
    // TotalSupervoxelProbability = Epsilon1 * ProbabilityFromNaturalDistribution + Epsilon2 * ProbabilityFromOutilers
    // Epsilon1 + Epsilon2 = 1
    // TotalSupervoxelProbability has mass 1
    
    for (svItr = supervoxel.begin(); svItr != supervoxel.end(); ++svItr) {
        
        SData::Ptr supervoxel = svItr->second;
        SData::VoxelVectorPtr voxels = supervoxel->getVoxelAVector();
        
        Eigen::Vector4f supervoxelCentroid = supervoxel->getCentroid();
        Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();
        
        
        float probabilityOutliers =  voxels->size() * svr_util::cube<float>(vr) * PROBABILITY_OUTLIERS_SUPERVOXEL; // Epsilon2
        
        typename SData::VoxelVector::iterator voxelItr;
        for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {
    
            float ax, bx, ay, by, az, bz;
            float probabilityFromND = svr_util::calculateApproximateIntegralForVoxel(ax, bx, ay, by, az, bz, supervoxelCovariance, supervoxelCentroid);
            
            int voxelSize = (*voxelItr)->getIndexVector()->size();
            supervoxelPointCount += voxelSize;
            supervoxelNormal += (*voxelItr)->getNormal();
        }

        
    }
    
}

void
SupervoxelRegistration::createSuperVoxelMappingForScan2 () {

	AdjacencyOctreeT adjTree1 = supervoxelClustering.getOctreeeAdjacency();

	AdjacencyOctreeT adjTree2;
	adjTree2.reset (new typename SupervoxelClusteringT::OctreeAdjacencyT(vr));
	adjTree2->setInputCloud(B);
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

	for (scanItr = B->begin(); scanItr != B->end(); ++scanItr, ++scanCounter) {

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
			PointT p = B->at(*indexItr);
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
					SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
					supervoxel->setLabel(label);
					supervoxel->getVoxelBVector()->push_back(voxel);
					// Add SV to SVMapping
					supervoxelMap.insert(std::pair<uint, typename SData::Ptr>(label, supervoxel));
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

        Eigen::Vector4f supervoxelCentroid;
        VData::ScanIndexVector supervoxelScanIndices;
        
		for (voxelItr = voxels->begin(); voxelItr != voxels->end(); ++voxelItr) {

			VData::Ptr voxel = *voxelItr;
			typename VData::ScanIndexVectorPtr indexVector = voxel->getIndexVector();
            
            supervoxelScanIndices.insert(supervoxelScanIndices.end(), indexVector -> begin(), indexVector -> end());
			PointT voxelCentroid;

			double xv(0), yv(0), zv(0), rv(0), gv(0), bv(0);
			for (indexItr = indexVector->begin(); indexItr != indexVector->end(); ++indexItr) {
				PointT pv = B->at(*indexItr);

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

//	calculateSupervoxelScanBData();

//	double mi = calculateMutualInformation();

	double mi = 0;
	//	cout << "MI Function Called with refreshed values" << mi << endl;

	return -mi;
}

Eigen::Affine3d
SupervoxelRegistration::optimize() {

	MI_Opti_Data* mod = new MI_Opti_Data();
	mod->scan1 = A;
	mod->scan2 = B;
	mod->svMap = &supervoxelMap;

	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;

	//	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
	gsl_multimin_fminimizer *s = NULL;

	gsl_vector *ss, *baseX;
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

	baseX = gsl_vector_alloc (6);
	gsl_vector_set (baseX, 0, 0);
	gsl_vector_set (baseX, 1, 0);
	gsl_vector_set (baseX, 2, 0);
	gsl_vector_set (baseX, 3, 0);
	gsl_vector_set (baseX, 4, 0);
	gsl_vector_set (baseX, 5, 0);


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
	createSuperVoxelMappingForScan2();

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

		cout << svLabel << '\t' << "A: " << counterA << '\t' << "B: " << counterB << endl;
	}

	showPointCloud(newCloud);
}







