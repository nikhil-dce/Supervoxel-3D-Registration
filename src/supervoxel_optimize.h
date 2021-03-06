
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

#include <gsl/gsl_multimin.h>
#include <pcl/common/transforms.h>

#define N_DIMEN 6
#define NSQRT_EXP 0.60653066

namespace svr_optimize {

struct svr_opti_data {
	svr::SVMap* svMap;
	svr::PointCloudT::Ptr scan2;
	Eigen::Affine3d t;
	bool approx;
};

class svrOptimize {

public:

	svrOptimize();

	~svrOptimize();

	void setOptimizeData(svr_opti_data data) {
		opt_data = data;
	}

	void optimizeUsingGaussNewton(Eigen::Affine3d& resultantTransform, float& cost);

	void computeCost(double& cost, svr::PointCloudT::Ptr transformedScan);

	void computeCostGradientHessian(double &cost, Eigen::VectorXf& g, Eigen::MatrixXf& H, svr::PointCloudT::Ptr transformedScan);

private:

//	double inline computePointFCost (svr::PointT p, Eigen::Vector4f& mean, Eigen::Matrix3f& covarianceInverse, double d1, double d2);
//	double computeSupervoxelFCost (SData::Ptr supervoxel, svr::PointCloudT::Ptr scan);
//	double f(Eigen::VectorXf& pose);
//
//	void inline computePointDfCost (svr::PointT p, Eigen::Vector4f& mean, Eigen::Matrix3f& covarianceInv, double d1, double d2, Eigen::VectorXf& df, bool appx);
//	void computeSupervoxelDfCost (SData::Ptr supervoxel, svr::PointCloudT::Ptr scan, const Eigen::VectorXf& pose, Eigen::VectorXf& df, bool appx);
//	void df (const Eigen::VectorXf& pose);
//
//	void computeSupervoxelFdf(SData::Ptr supervoxel, svr::PointCloudT::Ptr scan, const Eigen::VectorXf& pose, Eigen::VectorXf& df, double* fCost, bool appx);
//	void fdf (const Eigen::VectorXf& pose, double* fcost, Eigen::VectorXf& df);

	svr_opti_data opt_data;

};

//// Magnusson
//double inline computePointFCost (svr::PointT p, Eigen::Vector4f& mean, Eigen::Matrix3f& covarianceInverse, double d1, double d2) {
//
//	double c;
//
//	Eigen::Vector3f X;
//	X << p.x, p.y, p.z;
//
//	Eigen::Vector3f U;
//	U << mean(0), mean(1), mean(2);
//
//	float power = (X-U).transpose() * covarianceInverse * (X-U);
//
//	c = d1 * exp( -d2 * power / 2);
//
//	return c;
//}
//
//double computeSupervoxelFCost(SData::Ptr supervoxel, svr::PointCloudT::Ptr scan) {
//
//	double cost = 0;
//
//	Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();
//	Eigen::Vector4f supervoxelMean = supervoxel->getCentroid();
//
//	double d1 = supervoxel->getD1();
//	double d2 = supervoxel->getD2();
//	Eigen::Matrix3f supervoxelCovarianceInverse = supervoxel->getCovarianceInverse();
//
//	SData::ScanIndexVectorPtr indices = supervoxel->getScanBIndexVector();
//	SData::ScanIndexVector::iterator itr;
//
//	for (itr = indices->begin(); itr != indices->end(); ++itr) {
//		int index = *itr;
//		svr::PointT p = scan->at(index);
//		cost += computePointFCost(p, supervoxelMean, supervoxelCovarianceInverse, d1, d2);
//	}
//
////	if (cost < 0 || epsilon1 <= 0 || epsilon2 <= 0) {
////		cout << " Points in A " << supervoxel->getPointACount() << endl;
////		cout << " Label: " << supervoxel->getLabel() << endl;
////		cout << " Cost: " << cost << endl;
////		cout << " Covariance: " << endl << supervoxelCovariance << endl;
////		cout << " Epsilons: " << epsilon1 << ' ' << epsilon2 << endl;
////	}
//	return cost;
//}
//
//double f (const gsl_vector *pose, void* params) {
//
////	if (svr::_SVR_DEBUG_)
////		cout << "Calculating f" << endl;
//
//	clock_t start = svr_util::getClock();
//
//	// Initialize All Data
//	double x, y, z, roll, pitch ,yaw;
//	x = gsl_vector_get(pose, 0);
//	y = gsl_vector_get(pose, 1);
//	z = gsl_vector_get(pose, 2);
//	roll = gsl_vector_get(pose, 3);
//	pitch = gsl_vector_get(pose, 4);
//	yaw = gsl_vector_get(pose, 5);
//
//	svr_opti_data* optiData = (svr_opti_data*) params;
//
//	svr::PointCloudT::Ptr scan2 = optiData->scan2;
//	svr::PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());
//	svr::SVMap* SVMapping = optiData->svMap;
//
//	// Create Transformation
//	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
//	transform.translation() << x,y,z;
//	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
//	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
//	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
//
//	// Transform point cloud
//	pcl::transformPointCloud(*scan2, *transformedScan2, transform);
//
//	double cost = 0;
//	svr::SVMap::iterator svItr;
//	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
//		SData::Ptr supervoxel = svItr->second;
//		cost += computeSupervoxelFCost(supervoxel, transformedScan2);
//	}
//
////	clock_t end = svr_util::getClock();
////	if (svr::_SVR_DEBUG_)
////		std::cout << "f: " << svr_util::getClockTime(start, end) << std::endl;
//
//	return cost;
//}
//
//void inline computePointDfCost (svr::PointT p, Eigen::Vector4f& mean, Eigen::Matrix3f& covarianceInv, double d1, double d2, const gsl_vector* pose, gsl_vector* df, bool appx) {
//
//	Eigen::Vector3f X;
//	X << p.x, p.y, p.z;
//
//	Eigen::Vector3f U;
//	U << mean(0), mean(1), mean(2);
//
//	double x, y, z, roll, pitch ,yaw;
//	x = gsl_vector_get(pose, 0);
//	y = gsl_vector_get(pose, 1);
//	z = gsl_vector_get(pose, 2);
//	roll = gsl_vector_get(pose, 3);
//	pitch = gsl_vector_get(pose, 4);
//	yaw = gsl_vector_get(pose, 5);
//
//	Eigen::MatrixXf Jacobian (3,6);
//
//	if (appx) {
//
//		Jacobian << 1,0,0,0,X(2),-X(1),
//				0,1,0,-X(2),0,X(0),
//				0,0,1,X(1),-X(0),0;
//
//	} else {
//		double cx = cos(roll);
//		double sx = sin(roll);
//		double cy = cos(pitch);
//		double sy = sin(pitch);
//		double cz = cos(yaw);
//		double sz = sin(yaw);
//
//		double a = X(0) * (-sx*sz + cx*sy*cz)	+	X(1) * (-sx*cz - cx*sy*sz) 	+	X(2) * (-cx*cy);
//		double b = X(0) * (cx*sz + sx*sy*cz) 	+	X(1) * (-sx*sy*sz + cx*cz) 	+ 	X(2) * (-sx*cy);
//		double c = X(0) * (-sy*cz) 			 	+ 	X(1) * (sy*sz) 				+ 	X(2) * cy;
//		double d = X(0) * (sx*cy*cz) 		 	+	X(1) * (-sx*cy*sz) 			+ 	X(2) * (sx*sy);
//		double e = X(0) * (-cx*cy*cz)        	+  	X(1) * (cx*cy*sz) 			+ 	X(2) * (-cx*sy);
//		double f = X(0) * (-cy*sz) 			 	+ 	X(1) * (-cy*cz);
//		double g = X(0) * (cx*cz - sx*sy*sz) 	+ 	X(1) * (-cx*sz - sx*sy*cz);
//		double h = X(0) * (sx*cz + cx*sy*sz) 	+ 	X(1) * (cx*sy*cz - sx*sz);
//
//		Jacobian << 1,0,0,0,c,f,
//				0,1,0,a,d,g,
//				0,0,1,b,e,h;
//	}
//
//	float power = (X-U).transpose() * covarianceInv * (X-U);
//	power = -d2 * power / 2;
//
//	Eigen::MatrixXf r (1,6);
//	r = (X-U).transpose() * covarianceInv * Jacobian;
//	r = -d1 * d2 * r * exp(power);
//	double r0 = r(0);
//	double r1 = r(1);
//	double r2 = r(2);
//	double r3 = r(3);
//	double r4 = r(4);
//	double r5 = r(5);
//
//	gsl_vector* pointDf;
//	pointDf = gsl_vector_alloc(6);
//
//	gsl_vector_set(pointDf, 0, r0);
//	gsl_vector_set(pointDf, 1, r1);
//	gsl_vector_set(pointDf, 2, r2);
//	gsl_vector_set(pointDf, 3, r3);
//	gsl_vector_set(pointDf, 4, r4);
//	gsl_vector_set(pointDf, 5, r5);
//
//	gsl_vector_add(df, pointDf);
//}
//
//void computeSupervoxelDfCost(SData::Ptr supervoxel, svr::PointCloudT::Ptr scan, const gsl_vector* pose, gsl_vector* df, bool appx) {
//
//	Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();
//	Eigen::Vector4f supervoxelMean = supervoxel->getCentroid();
//	double d2 = supervoxel->getD2();
//	double d1 = supervoxel->getD1();
//	Eigen::Matrix3f supervoxelCovarianceInverse = supervoxel->getCovarianceInverse();
//
//	SData::ScanIndexVectorPtr indices = supervoxel->getScanBIndexVector();
//	SData::ScanIndexVector::iterator itr;
//
//	for (itr = indices->begin(); itr != indices->end(); ++itr) {
//		int index = *itr;
//		svr::PointT p = scan->at(index);
//		computePointDfCost(p, supervoxelMean, supervoxelCovarianceInverse, d1, d2, pose, df, appx);
//	}
//}
//
//// Magnusson
//void df (const gsl_vector *pose, void *params, gsl_vector *df) {
//
//	gsl_vector_set_zero(df);
//
////	if (svr::_SVR_DEBUG_)
////		cout << "Calculating df" << endl;
//
//	clock_t start = svr_util::getClock();
//
//	// Initialize All Data
//	double x, y, z, roll, pitch ,yaw;
//	x = gsl_vector_get(pose, 0);
//	y = gsl_vector_get(pose, 1);
//	z = gsl_vector_get(pose, 2);
//	roll = gsl_vector_get(pose, 3);
//	pitch = gsl_vector_get(pose, 4);
//	yaw = gsl_vector_get(pose, 5);
//
//	svr_opti_data* optiData = (svr_opti_data*) params;
//
//	svr::PointCloudT::Ptr scan2 = optiData->scan2;
//	svr::PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());
//	svr::SVMap* SVMapping = optiData->svMap;
//	bool appx = optiData->approx;
//
//	// Create Transformation
//	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
//	transform.translation() << x,y,z;
//	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
//	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
//	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
//
//	// Transform point cloud
//	pcl::transformPointCloud(*scan2, *transformedScan2, transform);
//
//	svr::SVMap::iterator svItr;
//	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
//		SData::Ptr supervoxel = svItr->second;
//		computeSupervoxelDfCost(supervoxel, transformedScan2, pose, df, appx);
//	}
//
//	clock_t end = svr_util::getClock();
//
////	if (svr::_SVR_DEBUG_)
////		std::cout << "df: " << svr_util::getClockTime(start, end) << std::endl;
//
//}
//
//void computeSupervoxelFdf(SData::Ptr supervoxel, svr::PointCloudT::Ptr scan, const gsl_vector* pose, gsl_vector* df, double* fCost, bool appx) {
//
//	Eigen::Matrix3f supervoxelCovariance = supervoxel->getCovariance();
//	Eigen::Vector4f supervoxelMean = supervoxel->getCentroid();
//
//	double d1 = supervoxel->getD1();
//	double d2 = supervoxel->getD2();
//	Eigen::Matrix3f supervoxelCovarianceInverse = supervoxel->getCovarianceInverse();
//
//	SData::ScanIndexVectorPtr indices = supervoxel->getScanBIndexVector();
//	SData::ScanIndexVector::iterator itr;
//
//	for (itr = indices->begin(); itr != indices->end(); ++itr) {
//		int index = *itr;
//		svr::PointT p = scan->at(index);
//		computePointDfCost(p, supervoxelMean, supervoxelCovarianceInverse, d1, d2, pose, df, appx);
//		*fCost += computePointFCost(p, supervoxelMean, supervoxelCovarianceInverse, d1, d2);
//	}
//
//}
//
//void fdf (const gsl_vector *pose, void *params, double *fCost, gsl_vector *df) {
//
//	gsl_vector_set_zero(df);
//
////	if (svr::_SVR_DEBUG_)
////		cout << "Calculating fdf" << endl;
//
//	clock_t start = svr_util::getClock();
//
//	// Initialize All Data
//	double x, y, z, roll, pitch ,yaw;
//	x = gsl_vector_get(pose, 0);
//	y = gsl_vector_get(pose, 1);
//	z = gsl_vector_get(pose, 2);
//	roll = gsl_vector_get(pose, 3);
//	pitch = gsl_vector_get(pose, 4);
//	yaw = gsl_vector_get(pose, 5);
//
//	svr_opti_data* optiData = (svr_opti_data*) params;
//
//	svr::PointCloudT::Ptr scan2 = optiData->scan2;
//	svr::PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());
//	svr::SVMap* SVMapping = optiData->svMap;
//	bool appx = optiData->approx;
//
//	// Create Transformation
//	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
//	transform.translation() << x,y,z;
//	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
//	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
//	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
//
//	// Transform point cloud
//	pcl::transformPointCloud(*scan2, *transformedScan2, transform);
//
//	svr::SVMap::iterator svItr;
//	for (svItr = SVMapping->begin(); svItr != SVMapping->end(); ++svItr) {
//		SData::Ptr supervoxel = svItr->second;
//		computeSupervoxelFdf(supervoxel, transformedScan2, pose, df, fCost, appx);
//	}
//
//	clock_t end = svr_util::getClock();
//
////	if (svr::_SVR_DEBUG_) {
////		cout << "fdf " << svr_util::getClockTime(start, end) << std::endl;
////		cout << "F: " << *fCost << std::endl;
////		std::cout << "df: " << std::endl;
////		std::cout << gsl_vector_get(df, 0) << std::endl;
////		std::cout << gsl_vector_get(df, 1) << std::endl;
////		std::cout << gsl_vector_get(df, 2) << std::endl;
////		std::cout << gsl_vector_get(df, 3) << std::endl;
////		std::cout << gsl_vector_get(df, 4) << std::endl;
////		std::cout << gsl_vector_get(df, 5) << std::endl;
////	}
//
//}
//
//const char* Status(int status) { return gsl_strerror(status); }
//
//void
//optimize(svr_opti_data opt_data, Eigen::Affine3d& resultantTransform, float& cost) {
//
//	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
////	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_conjugate_fr;
//	gsl_multimin_fdfminimizer *gsl_minimizer = NULL;
//	Eigen::Affine3d last_transform = opt_data.t;
//
//	double x,y,z,roll,pitch,yaw;
//
//	svr_util::transform_get_translation_from_affine(last_transform, &x, &y, &z);
//	svr_util::transform_get_rotation_from_affine(last_transform, &roll, &pitch, &yaw);
//
//	gsl_vector *baseX;
//
//	bool debug = true;
//	int max_iter = 20;
//	double line_search_tol = .01;
//	double gradient_tol = 1e-2;
//	double step_size = 1.;
//
//	// set up the gsl function_fdf struct
//	gsl_multimin_function_fdf function;
//	function.f = f;
//	function.df = df;
//	function.fdf = fdf;
//	function.n = N_DIMEN;
//	function.params = &opt_data;
//
//	size_t iter = 0;
//	int status;
//
//	baseX = gsl_vector_alloc (6);
//	gsl_vector_set (baseX, 0, x);
//	gsl_vector_set (baseX, 1, y);
//	gsl_vector_set (baseX, 2, z);
//	gsl_vector_set (baseX, 3, roll);
//	gsl_vector_set (baseX, 4, pitch);
//	gsl_vector_set (baseX, 5, yaw);
//
//	gsl_minimizer = gsl_multimin_fdfminimizer_alloc (T, N_DIMEN);
//	gsl_multimin_fdfminimizer_set (gsl_minimizer, &function, baseX, step_size, line_search_tol);
//
//	status = GSL_CONTINUE;
//	while (status == GSL_CONTINUE && iter < max_iter) {
//
//		iter++;
//		status = gsl_multimin_fdfminimizer_iterate (gsl_minimizer);
//
//		if(debug)
//			std::cout << iter << "\t\t" << gsl_minimizer->f << "\t\t" << Status(status) << std::endl;
//
//		if (status)
//			break;
//
//		status = gsl_multimin_test_gradient (gsl_minimizer->gradient, gradient_tol);
//	}
//
//
//	if (status == GSL_SUCCESS || iter == max_iter) {
//
//		if (svr::_SVR_DEBUG_) {
//			std::cout << "Cost: " << gsl_minimizer->f << " Iteration: " << iter << std::endl;
//			std::cout << "Converged to minimum at " << std::endl;
//		}
//
//		double tx = gsl_vector_get (gsl_minimizer->x, 0);
//		double ty = gsl_vector_get (gsl_minimizer->x, 1);
//		double tz = gsl_vector_get (gsl_minimizer->x, 2);
//		double roll = gsl_vector_get (gsl_minimizer->x, 3);
//		double pitch = gsl_vector_get (gsl_minimizer->x, 4);
//		double yaw = gsl_vector_get (gsl_minimizer->x, 5);
//
//		if (svr::_SVR_DEBUG_) {
//			std::cout << "Tx: " << tx << std::endl;
//			std::cout << "Ty: " << ty << std::endl;
//			std::cout << "Tz: " << tz << std::endl;
//			std::cout << "Roll: " << roll << std::endl;
//			std::cout << "Pitch: " << pitch << std::endl;
//			std::cout << "Yaw: " << yaw << std::endl;
//		}
//
//		//Eigen::Affine3d resultantTransform = Eigen::Affine3d::Identity();
//		resultantTransform = Eigen::Affine3d::Identity();
//		resultantTransform.translation() << tx, ty, tz;
//		resultantTransform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
//		resultantTransform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
//		resultantTransform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
//		cost = gsl_minimizer->f;
//
//		gsl_vector_free(baseX);
//		gsl_multimin_fdfminimizer_free(gsl_minimizer);
//
//		return;
//
//	} else if (status == GSL_ENOPROG) {
//
//		if (svr::_SVR_DEBUG_) {
//			std::cout << "Cost: " << gsl_minimizer->f << " Iteration: " << iter << std::endl;
//			std::cout << "Not making any progress " << std::endl;
//		}
//
//		double tx = gsl_vector_get (gsl_minimizer->x, 0);
//		double ty = gsl_vector_get (gsl_minimizer->x, 1);
//		double tz = gsl_vector_get (gsl_minimizer->x, 2);
//		double roll = gsl_vector_get (gsl_minimizer->x, 3);
//		double pitch = gsl_vector_get (gsl_minimizer->x, 4);
//		double yaw = gsl_vector_get (gsl_minimizer->x, 5);
//
//		if (svr::_SVR_DEBUG_) {
//			std::cout << "Tx: " << tx << std::endl;
//			std::cout << "Ty: " << ty << std::endl;
//			std::cout << "Tz: " << tz << std::endl;
//			std::cout << "Roll: " << roll << std::endl;
//			std::cout << "Pitch: " << pitch << std::endl;
//			std::cout << "Yaw: " << yaw << std::endl;
//		}
//
//		resultantTransform = Eigen::Affine3d::Identity();
//		resultantTransform.translation() << tx, ty, tz;
//		resultantTransform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
//		resultantTransform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
//		resultantTransform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));
//		cost = gsl_minimizer->f;
//
//		gsl_vector_free(baseX);
//		gsl_multimin_fdfminimizer_free(gsl_minimizer);
//
//		return;
//	}
//
//	if (svr::_SVR_DEBUG_)
//		std::cout << "Algorithm failed to converge" << std::endl;
//
//	gsl_vector_free(baseX);
//	gsl_multimin_fdfminimizer_free(gsl_minimizer);
//
//	resultantTransform = Eigen::Affine3d::Identity();
//	cost = 0;
//}

}


#endif /* svr_optimize_h */
