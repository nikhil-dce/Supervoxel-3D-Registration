#include "supervoxel_mapping.hpp"

VData::VData() {
	centroidCloudIndex = 0;
	indexVector.reset(new ScanIndexVector());
}

VData::~VData() {

}

SData::SData() {
	label = -1;
	pointACount = 0;
	pointBCount = 0;
	voxelsA.reset(new VoxelVector());
	voxelsB.reset(new VoxelVector());
}

SData::~SData() {

}

SuperVoxelMappingHelper::SuperVoxelMappingHelper(unsigned int label) {

	// Variance Features
	this->varianceXCodeA = 0;
	this->varianceXCodeB = 0;
	this->varianceYCodeA = 0;
	this->varianceYCodeB = 0;
	this->varianceZCodeA = 0;
	this->varianceZCodeB = 0;

	// Centroid Features
	this->centroidCodeA = 0;
	this->centroidCodeB = 0;

	// Normal Features
	this->normalCodeA = 0;
	this->normalCodeB = 0;

	this->label = label;
	this->scanACount = 0;
	this->scanBCount = 0;
	SimpleVoxelMapPtr p;
	voxelMap.reset(new typename SuperVoxelMappingHelper::SimpleVoxelMap());
}

SuperVoxelMappingHelper::~SuperVoxelMappingHelper() {

}

typename SuperVoxelMappingHelper::SimpleVoxelMapPtr
SuperVoxelMappingHelper::getVoxels() {
	return voxelMap;
}

SimpleVoxelMappingHelper::~SimpleVoxelMappingHelper() {}

SimpleVoxelMappingHelper::SimpleVoxelMappingHelper() {
	indicesAPtr.reset(new typename SimpleVoxelMappingHelper::ScanIndexVector());
	indicesBPtr.reset(new typename SimpleVoxelMappingHelper::ScanIndexVector());
	idxA = 0;
	idxB = 0;
};

typename SimpleVoxelMappingHelper::ScanIndexVectorPtr
SimpleVoxelMappingHelper::getScanBIndices() {
	return indicesBPtr;
}

typename SimpleVoxelMappingHelper::ScanIndexVectorPtr
SimpleVoxelMappingHelper::getScanAIndices() {
	return indicesAPtr;
}
