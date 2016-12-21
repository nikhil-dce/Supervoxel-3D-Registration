
//
//  svr_optimize.h
//  supervoxel_EM
//
//  Created by Nikhil on 21/12/16.
//
//

#ifndef svr_optimize_h
#define svr_optimize_h

#include "supervoxel_registration.h"
#include "supervoxel_util.hpp"

#include <gsl/gsl_multimin.h>
#include <pcl/common/transforms.h>

#define N_DIMEN 6
#define NSQRT_EXP 0.60653066

namespace svr_optimize {

struct svr_opti_data {
	svr::SVMap* svMap;
	svr::PointCloudT::Ptr scan1;
	svr::PointCloudT::Ptr scan2;
	Eigen::Affine3d t;
};

// Magnusson
double inline computePointFCost (svr::PointT p, Eigen::Vector4f mean, Eigen::Matrix3f covariance, double epsilon1, double epsilon2) {

	double c;

	double d3 = -log(epsilon2);
	double d1 = -log(epsilon1 + epsilon2) -d3;
	double d2 = -2 * log( (-log(epsilon1 * NSQRT_EXP + epsilon2) - d3) / d1);

	Eigen::Vector3f X;
	X << p.x, p.y, p.z;

	Eigen::Vector3f U;
	U << mean(0), mean(1), mean(2);

	float power = (X-U).transpose() * covariance.inverse() * (X-U);

	c = -d1 * exp( -d2 * power / 2);

	return c;
}

double computeSupervoxelFCost(SData::Ptr supervoxel, svr::PointCloudT::Ptr scan) {

	double cost = 0;

	double epsilon1 = supervoxel->getEpsilon1();
	double epsilon2 = supervoxel->getEpsilon2();

	Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();
	Eigen::Vector4f supervoxelMean = supervoxel->getCentroid();

	SData::VoxelVectorPtr voxels = supervoxel->getVoxelBVector();
	SData::VoxelVector::iterator vxlItr;

	for (vxlItr = voxels->begin(); vxlItr != voxels->end(); ++vxlItr) {

		VData::Ptr voxel = *vxlItr;

		VData::ScanIndexVectorPtr indices = voxel->getIndexVector();
		VData::ScanIndexVector::iterator itr;

		for (itr = indices->begin(); itr != indices->end(); ++itr) {
			int index = *itr;
			svr::PointT p = scan->at(index);
			cost += computePointFCost(p, supervoxelMean, supervoxelCovariance, epsilon1, epsilon2);
		}

	}

	if (cost < 0 || epsilon1 <= 0 || epsilon2 <= 0) {
		cout << " Points in A " << supervoxel->getPointACount() << endl;
		cout << " Label: " << supervoxel->getLabel() << endl;
		cout << " Cost: " << cost << endl;
		cout << " Covariance: " << endl << supervoxelCovariance << endl;
		cout << " Epsilons: " << epsilon1 << ' ' << epsilon2 << endl;
	}
	return cost;
}

double f (const gsl_vector *pose, void* params) {

	// Initialize All Data
	double x, y, z, roll, pitch ,yaw;
	x = gsl_vector_get(pose, 0);
	y = gsl_vector_get(pose, 1);
	z = gsl_vector_get(pose, 2);
	roll = gsl_vector_get(pose, 3);
	pitch = gsl_vector_get(pose, 4);
	yaw = gsl_vector_get(pose, 5);

	cout << "Pose Test: " << endl;
	cout << gsl_vector_get(pose, 0) << endl;
	cout << gsl_vector_get(pose, 1) << endl;
	cout << gsl_vector_get(pose, 2) << endl;
	cout << gsl_vector_get(pose, 3) << endl;
	cout << gsl_vector_get(pose, 4) << endl;
	cout << gsl_vector_get(pose, 5) << endl;


	svr_opti_data* optiData = (svr_opti_data*) params;

	svr::PointCloudT::Ptr scan1 = optiData->scan1;
	svr::PointCloudT::Ptr scan2 = optiData->scan2;
	svr::PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());

	svr::SVMap* SVMapping = optiData->svMap;

	// Create Transformation
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
	transform.translation() << x,y,z;
	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

	// Transform point cloud
	pcl::transformPointCloud(*scan2, *transformedScan2, transform);

	double cost = 0;
	svr::SVMap::iterator svItr;
	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
		SData::Ptr supervoxel = svItr->second;
		cost += computeSupervoxelFCost(supervoxel, transformedScan2);
	}

	std::cout << "f: " << cost << std::endl;
	return -cost;
}

void inline computePointDfCost (svr::PointT p, Eigen::Vector4f mean, Eigen::Matrix3f covariance, double epsilon1, double epsilon2, const gsl_vector* pose, gsl_vector* df) {

	double d3 = -log(epsilon2);
	double d1 = -log(epsilon1 + epsilon2) -d3;
	double d2 = -2 * log( (-log(epsilon1 * NSQRT_EXP + epsilon2) - d3) / d1);

	Eigen::Matrix3f covarianceInv = covariance.inverse();

	Eigen::Vector3f X;
	X << p.x, p.y, p.z;

	Eigen::Vector3f U;
	U << mean(0), mean(1), mean(2);

	double x, y, z, roll, pitch ,yaw;
	x = gsl_vector_get(pose, 0);
	y = gsl_vector_get(pose, 1);
	z = gsl_vector_get(pose, 2);
	roll = gsl_vector_get(pose, 3);
	pitch = gsl_vector_get(pose, 4);
	yaw = gsl_vector_get(pose, 5);

	double cx = cos(roll);
	double sx = sin(roll);
	double cy = cos(pitch);
	double sy = sin(pitch);
	double cz = cos(yaw);
	double sz = sin(yaw);

	double a = X(0) * (-sx*sz + cx*sy*cz) + X(1) * (-sx*cz - cx*sy*sz) + X(2) * (-cx*cy);
	double b = X(0) * (cx*sz + sx*sy*cz) + X(1) * (-sx*sy*sz + cx*cz) + X(2) * (-sx*cy);
	double c = X(0) * (-sy*cz) + X(1) * (sy*sz) + X(2) * cy;
	double d = X(0) * (sx*cy*cz) + X(1) * (-sx*cy*sz) + X(2) * (sx*sy);
	double e = X(0) * (-cx*cy*cz) + X(1) * (cx*cy*sz) + X(2) * (-cx*sy);
	double f = X(0) * (-cy*sz) + X(1) * (-cy*cz);
	double g = X(0) * (cx*cz - sx*sy*sz) + X(1) * (-cx*sz - sx*sy*cz);
	double h = X(0) * (sx*cz + cx*sy*sz) + X(1) * (cx*sy*cz - sx*sz);

	Eigen::MatrixXf Jacobian (3,6);
	Jacobian << 1,0,0,0,c,f,
				0,1,0,a,d,g,
				0,0,1,b,e,h;

	float power = (X-U).transpose() * covarianceInv * (X-U);
	power = -d2 * power / 2;

	Eigen::MatrixXf r (1,6);
	r = (X-U).transpose() * covarianceInv * Jacobian;
	r = d1 * d2 * r * exp(power);
	double r0 = r(0);
	double r1 = r(1);
	double r2 = r(2);
	double r3 = r(3);
	double r4 = r(4);
	double r5 = r(5);

	gsl_vector* pointDf;
	pointDf = gsl_vector_alloc(6);

	gsl_vector_set(pointDf, 0, r0);
	gsl_vector_set(pointDf, 1, r1);
	gsl_vector_set(pointDf, 2, r2);
	gsl_vector_set(pointDf, 3, r3);
	gsl_vector_set(pointDf, 4, r4);
	gsl_vector_set(pointDf, 5, r5);

	gsl_vector_add(df, pointDf);
}

void computeSupervoxelDfCost(SData::Ptr supervoxel, svr::PointCloudT::Ptr scan, const gsl_vector* pose, gsl_vector* df) {

	double epsilon1 = supervoxel->getEpsilon1();
	double epsilon2 = supervoxel->getEpsilon2();
	Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();
	Eigen::Vector4f supervoxelMean = supervoxel->getCentroid();

	SData::VoxelVectorPtr voxels = supervoxel->getVoxelBVector();
	SData::VoxelVector::iterator vxlItr;

	for (vxlItr = voxels->begin(); vxlItr != voxels->end(); ++vxlItr) {

		VData::Ptr voxel = *vxlItr;

		VData::ScanIndexVectorPtr indices = voxel->getIndexVector();
		VData::ScanIndexVector::iterator itr;

		for (itr = indices->begin(); itr != indices->end(); ++itr) {
			int index = *itr;
			svr::PointT p = scan->at(index);
			computePointDfCost(p, supervoxelMean, supervoxelCovariance, epsilon1, epsilon2, pose, df);
		}

	}
}

// Magnusson
void df (const gsl_vector *pose, void *params, gsl_vector *df) {

	gsl_vector_set_zero(df);

	// Initialize All Data
	double x, y, z, roll, pitch ,yaw;
	x = gsl_vector_get(pose, 0);
	y = gsl_vector_get(pose, 1);
	z = gsl_vector_get(pose, 2);
	roll = gsl_vector_get(pose, 3);
	pitch = gsl_vector_get(pose, 4);
	yaw = gsl_vector_get(pose, 5);

	svr_opti_data* optiData = (svr_opti_data*) params;

	svr::PointCloudT::Ptr scan1 = optiData->scan1;
	svr::PointCloudT::Ptr scan2 = optiData->scan2;
	svr::PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());

	svr::SVMap* SVMapping = optiData->svMap;

	// Create Transformation
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
	transform.translation() << x,y,z;
	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

	// Transform point cloud
	pcl::transformPointCloud(*scan2, *transformedScan2, transform);

	svr::SVMap::iterator svItr;
	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
		SData::Ptr supervoxel = svItr->second;
		computeSupervoxelDfCost(supervoxel, transformedScan2, pose, df);
	}

//	std::cout << "df: " << std::endl;
//	std::cout << gsl_vector_get(df, 0) << std::endl;
//	std::cout << gsl_vector_get(df, 1) << std::endl;
//	std::cout << gsl_vector_get(df, 2) << std::endl;
//	std::cout << gsl_vector_get(df, 3) << std::endl;
//	std::cout << gsl_vector_get(df, 4) << std::endl;
//	std::cout << gsl_vector_get(df, 5) << std::endl;
}


void fdf (const gsl_vector *pose, void *params, double *fCost, gsl_vector *df) {

	std::cout << "fdf" << std::endl;

	gsl_vector_set_zero(df);

	// Initialize All Data
	double x, y, z, roll, pitch ,yaw;
	x = gsl_vector_get(pose, 0);
	y = gsl_vector_get(pose, 1);
	z = gsl_vector_get(pose, 2);
	roll = gsl_vector_get(pose, 3);
	pitch = gsl_vector_get(pose, 4);
	yaw = gsl_vector_get(pose, 5);

	svr_opti_data* optiData = (svr_opti_data*) params;

	svr::PointCloudT::Ptr scan1 = optiData->scan1;
	svr::PointCloudT::Ptr scan2 = optiData->scan2;
	svr::PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());
	svr::SVMap* SVMapping = optiData->svMap;

	// Create Transformation
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
	transform.translation() << x,y,z;
	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

//	cout << transform.matrix() << endl;

//	Eigen::Matrix3f nmat;
//	nmat = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
//	    		   * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
//	    		   * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
//
//	cout << "M: " << endl << nmat << endl;

	// Transform point cloud
	pcl::transformPointCloud(*scan2, *transformedScan2, transform);

	svr::SVMap::iterator svItr;
	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
		SData::Ptr supervoxel = svItr->second;
		computeSupervoxelDfCost(supervoxel, transformedScan2, pose, df);
		*fCost += computeSupervoxelFCost(supervoxel, transformedScan2);
	}

	cout << "F: " << *fCost << endl;
	std::cout << "df: " << std::endl;
	std::cout << gsl_vector_get(df, 0) << std::endl;
	std::cout << gsl_vector_get(df, 1) << std::endl;
	std::cout << gsl_vector_get(df, 2) << std::endl;
	std::cout << gsl_vector_get(df, 3) << std::endl;
	std::cout << gsl_vector_get(df, 4) << std::endl;
	std::cout << gsl_vector_get(df, 5) << std::endl;
}

const char* Status(int status) { return gsl_strerror(status); }

Eigen::Affine3d
optimize(svr_opti_data opt_data) {

	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
	gsl_multimin_fdfminimizer *gsl_minimizer = NULL;
	Eigen::Affine3d last_transform = opt_data.t;

	double x,y,z,roll,pitch,yaw;

	svr_util::transform_get_translation_from_affine(last_transform, &x, &y, &z);
	svr_util::transform_get_rotation_from_affine(last_transform, &roll, &pitch, &yaw);

	gsl_vector *baseX;

	bool debug = true;
	int max_iter = 20;
	double line_search_tol = .01;
	double gradient_tol = 1e-2;
	double step_size = 1.;

	// set up the gsl function_fdf struct
	gsl_multimin_function_fdf function;
	function.f = f;
	function.df = df;
	function.fdf = fdf;
	function.n = N_DIMEN;
	function.params = &opt_data;

	size_t iter = 0;
	int status;

	baseX = gsl_vector_alloc (6);
	gsl_vector_set (baseX, 0, x);
	gsl_vector_set (baseX, 1, y);
	gsl_vector_set (baseX, 2, z);
	gsl_vector_set (baseX, 3, roll);
	gsl_vector_set (baseX, 4, pitch);
	gsl_vector_set (baseX, 5, yaw);

	gsl_minimizer = gsl_multimin_fdfminimizer_alloc (T, N_DIMEN);
	gsl_multimin_fdfminimizer_set (gsl_minimizer, &function, baseX, step_size, line_search_tol);

	status = GSL_CONTINUE;
	while (status == GSL_CONTINUE && iter < max_iter) {

		iter++;
		status = gsl_multimin_fdfminimizer_iterate (gsl_minimizer);

		if(debug)
			std::cout << iter << "\t\t" << gsl_minimizer->f << "\t\t" << Status(status) << std::endl;

		if (status)
			break;

		status = gsl_multimin_test_gradient (gsl_minimizer->gradient, gradient_tol);
	}


	if (status == GSL_SUCCESS) {

		cout << "Cost: " << gsl_minimizer->f << " Iteration: " << iter << endl;
		cout << "Converged to minimum at " << endl;

		double tx = gsl_vector_get (gsl_minimizer->x, 0);
		double ty = gsl_vector_get (gsl_minimizer->x, 1);
		double tz = gsl_vector_get (gsl_minimizer->x, 2);
		double roll = gsl_vector_get (gsl_minimizer->x, 3);
		double pitch = gsl_vector_get (gsl_minimizer->x, 4);
		double yaw = gsl_vector_get (gsl_minimizer->x, 5);

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

		gsl_vector_free(baseX);
		gsl_multimin_fdfminimizer_free(gsl_minimizer);

		return resultantTransform;
	}

	std::cout << "Algorithm failed to converge" << std::endl;
	gsl_vector_free(baseX);
	gsl_multimin_fdfminimizer_free(gsl_minimizer);

	return Eigen::Affine3d::Identity();
}

}


#endif /* svr_optimize_h */
