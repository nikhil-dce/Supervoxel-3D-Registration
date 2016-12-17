/*
 * supervoxel_mapping.hpp
 *
 *  Created on: 24-Nov-2016
 *      Author: root
 */

#ifndef SUPERVOXEL_MAPPING_HPP_
#define SUPERVOXEL_MAPPING_HPP_

#include <vector>
#include <boost/shared_ptr.hpp>
#include <pcl/point_types.h>
#include "supervoxel_cluster_search.h"
#include <pcl/octree/octree_key.h>
#include <string>

struct MOctreeKey {

	MOctreeKey(pcl::octree::OctreeKey octreeKey) {
		this->key = octreeKey;
	}

	bool operator<(const MOctreeKey& arg) const
	{
		if (this->key.x < arg.key.x)
			return true;
		else if (this->key.x == arg.key.x && this->key.y < arg.key.y)
			return true;
		else if (this->key.x == arg.key.x && this->key.y == arg.key.y && this->key.z < arg.key.z)
			return true;

		return false;
	}

	typename pcl::octree::OctreeKey key;
};

class VData {

public:

	typedef boost::shared_ptr<VData> Ptr;
	typedef std::vector<int> ScanIndexVector;
	typedef boost::shared_ptr<ScanIndexVector> ScanIndexVectorPtr;

	VData();
	~VData();

	ScanIndexVectorPtr
	getIndexVector() {
		return indexVector;
	}

	Eigen::Vector3f
	getNormal() {
		return normal;
	}

	void
	setNormal(Eigen::Vector3f n) {
		normal = n;
	}

	void
	setCentroid(pcl::PointXYZRGBA p) {
		centroid = p;
	}

	pcl::PointXYZRGBA
	getCentroid() {
		return centroid;
	}

	int
	getCentroidCloudIndex() {
		return centroidCloudIndex;
	}

	void
	setCentroidCloudIndex(int index) {
		centroidCloudIndex = index;
	}

private:

	ScanIndexVectorPtr indexVector;
	Eigen::Vector3f normal;
	pcl::PointXYZRGBA centroid;
	int centroidCloudIndex;
};

class SData {

public:

	typedef boost::shared_ptr<SData> Ptr;
	typedef std::vector<VData::Ptr> VoxelVector;
	typedef boost::shared_ptr<VoxelVector> VoxelVectorPtr;

	SData();
	~SData();

	void
	setLabel(int l) {
		label = l;
	}

	int
	getLabel() {
		return label;
	}

	VoxelVectorPtr
	getVoxelAVector() {
		return voxelsA;
	}

	VoxelVectorPtr
	getVoxelBVector() {
		return voxelsB;
	}

	Eigen::Vector3f
	getNormal() {
		return normal;
	}

	void
	setNormal(Eigen::Vector3f n) {
		normal = n;
	}

	void
	setCentroidA(pcl::PointXYZRGBA p) {
		centroidA = p;
	}

	pcl::PointXYZRGBA
	getCentroidA() {
		return centroidA;
	}

	void
	setCentroidB(pcl::PointXYZRGBA p) {
		centroidB = p;
	}

	pcl::PointXYZRGBA
	getCentroidB() {
		return centroidB;
	}

	void
	setPointACount(int count) {
		pointACount = count;
	}

	void
	setPointBCount(int count) {
		pointBCount = count;
	}

	int
	getPointACount() {
		return pointACount;
	}

	int
	getPointBCount() {
		return pointBCount;
	}

	int
	getVarianceXCodeA() {
		return varianceXCodeA;
	}

	void
	setVarianceXCodeA(int code) {
		varianceXCodeA = code;
	}

	int
	getVarianceXCodeB() {
		return varianceXCodeB;
	}

	void
	setVarianceXCodeB(int code) {
		varianceXCodeB = code;
	}

	std::string
	getVarianceXCodeAB() {
		return varianceXCodeAB;
	}

	void
	setVarianceXCodeAB(std::string code) {
		varianceXCodeAB = code;
	}

	int
	getVarianceYCodeA() {
		return varianceYCodeA;
	}

	void
	setVarianceYCodeA(int code) {
		varianceYCodeA = code;
	}

	int
	getVarianceYCodeB() {
		return varianceYCodeB;
	}

	void
	setVarianceYCodeB(int code) {
		varianceYCodeB = code;
	}

	std::string
	getVarianceYCodeAB() {
		return varianceYCodeAB;
	}

	void
	setVarianceYCodeAB(std::string code) {
		varianceYCodeAB = code;
	}

	int
	getVarianceZCodeA() {
		return varianceZCodeA;
	}

	void
	setVarianceZCodeA(int code) {
		varianceZCodeA = code;
	}

	int
	getVarianceZCodeB() {
		return varianceZCodeB;
	}

	void
	setVarianceZCodeB(int code) {
		varianceZCodeB = code;
	}

	std::string
	getVarianceZCodeAB() {
		return varianceZCodeAB;
	}

	void
	setVarianceZCodeAB(std::string code) {
		varianceZCodeAB = code;
	}

	int
	getCentroidCodeA() {
		return centroidCodeA;
	}

	void
	setCentroidCodeA(int code) {
		centroidCodeA = code;
	}

	int
	getCentroidCodeB() {
		return centroidCodeB;
	}

	void
	setCentroidCodeB(int code) {
		centroidCodeB = code;
	}

	std::string
	getCentroidCodeAB() {
		return centroidCodeAB;
	}

	void
	setCentroidCodeAB(std::string code) {
		centroidCodeAB = code;
	}

	void
	clearScanBMapping() {
		voxelsB->clear();
		pointBCount = 0;
	}

	void
	clearScanBData() {
		// clear variance x
		varianceXCodeB = 0;
		varianceXCodeAB = "";

		// clear variance y
		varianceYCodeB = 0;
		varianceYCodeAB = "";

		// clear variance z
		varianceZCodeB = 0;
		varianceZCodeAB = "";

		centroidCodeB = 0;
		centroidCodeAB = "";

		centroidB.x = 0;
		centroidB.y = 0;
		centroidB.z = 0;
		centroidB.r = 0;
		centroidB.g = 0;
		centroidB.b = 0;
	}

private:

	int label;

	int centroidCodeA;
	int centroidCodeB;
	std::string centroidCodeAB;

	int varianceXCodeA;
	int varianceXCodeB;
	std::string varianceXCodeAB;

	int varianceYCodeA;
	int varianceYCodeB;
	std::string varianceYCodeAB;

	int varianceZCodeA;
	int varianceZCodeB;
	std::string varianceZCodeAB;

	VoxelVectorPtr voxelsA;
	VoxelVectorPtr voxelsB;
	int pointACount;
	int pointBCount;

	pcl::PointXYZRGBA centroidA;
	pcl::PointXYZRGBA centroidB;

	Eigen::Vector3f normal;
};

class SimpleVoxelMappingHelper {

public:

	typedef boost::shared_ptr<SimpleVoxelMappingHelper> Ptr;
	typedef std::vector<int> ScanIndexVector;
	typedef boost::shared_ptr<std::vector<int> > ScanIndexVectorPtr;

	SimpleVoxelMappingHelper();

	~SimpleVoxelMappingHelper();

	ScanIndexVectorPtr
	getScanAIndices();

	ScanIndexVectorPtr
	getScanBIndices();

	void
	setNormalA(typename pcl::PointNormal normal) {
		this->normalA = normal;
	}

	void
	setNormalB(typename pcl::PointNormal normal) {
		this->normalB = normal;
	}

	typename pcl::PointNormal
	getNormalA() {
		return this->normalA;
	}

	typename pcl::PointNormal
	getNormalB() {
		return this->normalB;
	}

	pcl::PointXYZRGBA
	getCentroidA() {
		return centroidA;
	}

	pcl::PointXYZRGBA
	getCentroidB() {
		return centroidB;
	}

	void
	setCentroidA(pcl::PointXYZRGBA centroid) {
		this->centroidA = centroid;
	}

	void
	setCentroidB(pcl::PointXYZRGBA centroid) {
		this->centroidB = centroid;
	}

	void
	setIndexA(int index) {
		idxA = index;
	}

	int
	getIndexA() {
		return idxA;
	}

	void
	setIndexB(int index) {
		idxB = index;
	}

	int
	getIndexB() {
		return idxB;
	}

	void
	clearScanBData() {
		centroidB.x = 0;
		centroidB.y = 0;
		centroidB.z = 0;
		centroidB.r = 0;
		centroidB.g = 0;
		centroidB.b = 0;
		normalB.normal_x = 0;
		normalB.normal_y = 0;
		normalB.normal_z = 0;
		normalB.x = 0;
		normalB.y = 0;
		normalB.z = 0;
		indicesBPtr -> clear();
		idxB = 0;
	}

private:

	ScanIndexVectorPtr indicesAPtr;
	ScanIndexVectorPtr indicesBPtr;
	typename pcl::PointNormal normalA;
	typename pcl::PointNormal normalB;
	//	typename pcl::RGB rgbA;
	//	typename pcl::RGB rgbB;
	pcl::PointXYZRGBA centroidA;
	pcl::PointXYZRGBA centroidB;
	//	typename pcl::SupervoxelClustering<pcl::PointXYZRGBA>::LeafContainerT* leafPtr;
	int idxA;
	int idxB;

};

class SuperVoxelMappingHelper {

public:

	typedef typename pcl::octree::OctreeKey OctreeKeyT;
	//	typedef typename std::map<MOctreeKey, typename SimpleVoxelMappingHelper::Ptr> SimpleVoxelMap;
	typedef typename std::map<pcl::SupervoxelClustering<pcl::PointXYZRGBA>::LeafContainerT*, typename SimpleVoxelMappingHelper::Ptr> SimpleVoxelMap;
	typedef typename boost::shared_ptr<SimpleVoxelMap> SimpleVoxelMapPtr;
	typedef typename boost::shared_ptr<SuperVoxelMappingHelper> Ptr;

	SuperVoxelMappingHelper(unsigned int);

	~SuperVoxelMappingHelper();

	unsigned int
	getSuperVoxelLabel();

	void
	setNormalA(typename pcl::PointNormal normal) {
		this->centroidNormalA = normal;
	}

	void
	setNormalB(typename pcl::PointNormal normal) {
		this->centroidNormalB = normal;
	}

	typename pcl::PointNormal
	getNormalA() {
		return this->centroidNormalA;
	}

	typename pcl::PointNormal
	getNormalB() {
		return this->centroidNormalB;
	}

	typename pcl::RGB
	getrgbA() {
		return centroidRGBA;
	}

	typename pcl::RGB
	getrgbB() {
		return centroidRGBB;
	}

	void
	setrgbA(typename pcl::RGB rgb) {
		this->centroidRGBA = rgb;
	}

	void
	setrgbB(typename pcl::RGB rgb) {
		this->centroidRGBB = rgb;
	}

	SimpleVoxelMapPtr
	getVoxels();

	unsigned int
	getScanBCount() {
		return scanBCount;
	}

	unsigned int
	getScanACount() {
		return scanACount;
	}

	void
	setScanACount(unsigned int count) {
		this->scanACount = count;
	}

	void
	setScanBCount (unsigned int count) {
		this->scanBCount = count;
	}

	void
	clearScanBData() {

		// clear variance x
		varianceXCodeB = 0;
		varianceXCodeAB = "";

		// clear variance y
		varianceYCodeB = 0;
		varianceYCodeAB = "";

		// clear variance z
		varianceZCodeB = 0;
		varianceZCodeAB = "";

		// Clear Normal and centroid data
		centroidNormalB.x = 0;
		centroidNormalB.y = 0;
		centroidNormalB.z = 0;
		centroidNormalB.normal_x = 0;
		centroidNormalB.normal_y = 0;
		centroidNormalB.normal_z = 0;
		centroidNormalB.curvature = 0;

		// clear centroid code
		centroidCodeB = 0;
		centroidCodeAB = "";

		// clear normal code
		normalCodeB = 0;
		normalCodeAB = "";

		scanBCount = 0;

		SimpleVoxelMapPtr voxelMap = this->getVoxels();
		SimpleVoxelMap::iterator voxelItr = voxelMap->begin();

		for (; voxelItr != voxelMap->end(); ++voxelItr) {
			SimpleVoxelMappingHelper::Ptr voxel = voxelItr->second;
			voxel->clearScanBData();
		}

	}

	// Normal Codes

	std::string
	getNormalCodeAB() {
		return normalCodeAB;
	}

	int
	getNormalCodeA() {
		return normalCodeA;
	}

	int
	getNormalCodeB() {
		return normalCodeB;
	}

	void
	setNormalCodeAB(std::string codeAB) {
		this->normalCodeAB = codeAB;
	}

	void
	setNormalCodeA(int code) {
		this->normalCodeA = code;
	}

	void
	setNormalCodeB(int code) {
		this->normalCodeB = code;
	}

	// Centroid

	std::string
	getCentroidCodeAB() {
		return centroidCodeAB;
	}

	int
	getCentroidCodeA() {
		return centroidCodeA;
	}

	int
	getCentroidCodeB() {
		return centroidCodeB;
	}

	void
	setCentroidCodeAB(std::string codeAB) {
		this->centroidCodeAB = codeAB;
	}

	void
	setCentroidCodeA(int code) {
		this->centroidCodeA = code;
	}

	void
	setCentroidCodeB(int code) {
		this->centroidCodeB = code;
	}

	int
	getVarianceXCodeA() {
		return varianceXCodeA;
	}

	void
	setVarianceXCodeA(int code) {
		varianceXCodeA = code;
	}

	int
	getVarianceXCodeB() {
		return varianceXCodeB;
	}

	void
	setVarianceXCodeB(int code) {
		varianceXCodeB = code;
	}

	std::string
	getVarianceXCodeAB() {
		return varianceXCodeAB;
	}

	void
	setVarianceXCodeAB(std::string code) {
		varianceXCodeAB = code;
	}

	int
	getVarianceYCodeA() {
		return varianceYCodeA;
	}

	void
	setVarianceYCodeA(int code) {
		varianceYCodeA = code;
	}

	int
	getVarianceYCodeB() {
		return varianceYCodeB;
	}

	void
	setVarianceYCodeB(int code) {
		varianceYCodeB = code;
	}

	std::string
	getVarianceYCodeAB() {
		return varianceYCodeAB;
	}

	void
	setVarianceYCodeAB(std::string code) {
		varianceYCodeAB = code;
	}

	int
	getVarianceZCodeA() {
		return varianceZCodeA;
	}

	void
	setVarianceZCodeA(int code) {
		varianceZCodeA = code;
	}

	int
	getVarianceZCodeB() {
		return varianceZCodeB;
	}

	void
	setVarianceZCodeB(int code) {
		varianceZCodeB = code;
	}

	std::string
	getVarianceZCodeAB() {
		return varianceZCodeAB;
	}

	void
	setVarianceZCodeAB(std::string code) {
		varianceZCodeAB = code;
	}

private :

	int varianceXCodeA;
	int varianceXCodeB;
	std::string varianceXCodeAB;

	int varianceYCodeA;
	int varianceYCodeB;
	std::string varianceYCodeAB;

	int varianceZCodeA;
	int varianceZCodeB;
	std::string varianceZCodeAB;

	int centroidCodeA;
	int centroidCodeB;
	std::string centroidCodeAB;

	int normalCodeA;
	int normalCodeB; // will change in every iteration during optimization
	std::string normalCodeAB;

	unsigned int scanBCount;
	unsigned int scanACount;

	unsigned int label;
	typename pcl::PointNormal centroidNormalA;
	typename pcl::PointNormal centroidNormalB;
	typename pcl::RGB centroidRGBA;
	typename pcl::RGB centroidRGBB;
	SimpleVoxelMapPtr voxelMap;
};

#endif /* SUPERVOXEL_MAPPING_HPP_ */
