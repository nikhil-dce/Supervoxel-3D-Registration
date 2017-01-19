#include "supervoxel_optimize.h"

using namespace svr_optimize;

svrOptimize::svrOptimize() {

}

svrOptimize::~svrOptimize() {

}

// Gauss Newton Optimization
void svrOptimize::optimizeUsingGaussNewton(Eigen::Affine3d& resultantTransform, float& cost) {

	svr::PointCloudT::Ptr scan2 = opt_data.scan2;
	svr::SVMap* svMap = opt_data.svMap;
	Eigen::Affine3d last_transform = opt_data.t;

	double x,y,z,roll,pitch,yaw;

	svr_util::transform_get_translation_from_affine(last_transform, &x, &y, &z);
	svr_util::transform_get_rotation_from_affine(last_transform, &roll, &pitch, &yaw);

	int maxIteration = 20, iteration = 0;
	bool debug = true, converged = false;
	double tol = 1e-4, stepSize = 1., poseRotTol = 1e-15, poseTransTol = 1e-10;
	float lambda = 1e-3;

	svr::SVMap::iterator svMapItr;
	SData::ScanIndexVector::iterator pItr;
	SData::ScanIndexVectorPtr indexVector;
	Eigen::Affine3d iterationTransform;
	svr::PointCloudT::Ptr transformedScan =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());
	double currentCost = 0;
	Eigen::MatrixXf H(6,6);
	Eigen::VectorXf g(6,1);


	while (!converged && iteration < maxIteration) {

		iteration++;
		iterationTransform = Eigen::Affine3d::Identity();
		iterationTransform.translation() << x,y,z;
		iterationTransform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
		iterationTransform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
		iterationTransform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

		// transform Scan
		pcl::transformPointCloud(*scan2, *transformedScan, iterationTransform);

		// Calculate Hessian and Gradient at last pose
		H.setZero();
		g.setZero();
		currentCost = 0;

		computeCostGradientHessian(currentCost, g, H, transformedScan);

		bool converged = true;
		for (int i = 0; i < 6; i++) {

			double gdiff = fabs((double)g(i));

			if (gdiff > tol) {
				converged = false;
				break;
			}
		}

		std::cout << "Iteration: " << iteration << std::endl;
		std::cout << "Current Cost: " << currentCost << std::endl;
		std::cout << "Gradient: " << std::endl << g << std::endl;

		if (converged) {
			resultantTransform = iterationTransform;
			cost = currentCost;
			return;
		}

		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				H(j,i) = H(i,j);
			}
		}

		std::cout << "Hessian: " << std::endl << H << std::endl;

		Eigen::MatrixXf A = H;


		bool iterationComplete = false;
		Eigen::VectorXf poseStep(6,1);

		while (!iterationComplete) {

			// till cost doesnt decrease
			// switch between gauss newton and gradient descent
			std::cout << "Lambda: " << lambda << std::endl;

			A = H;
			for (int i = 0; i < 6; i++) {
				A(i,i) *= (lambda + 1);
			}

			//			std::cout << "A: " << std::endl << A;
			//			std::cout << std::endl;
			poseStep = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g);
			poseStep = stepSize * poseStep;
			//		poseStep = stepSize * g;
			std::cout << "Pose Step: " << std::endl << poseStep << std::endl;

			//

			bool iterationProgressing = false;

			for (int k=0; k<6; k++) {
				double poseDiff = fabs((double)poseStep(k));
				if (k < 3 && poseDiff > poseTransTol) {
					iterationProgressing = true;
					break;
				} else if (k > 3 && poseDiff > poseRotTol) {
					iterationProgressing = true;
					break;
				}
			}

			if (!iterationProgressing) {
				// not converging
				std::cout << "Pose step too small" << std::endl;
				resultantTransform = iterationTransform;
				cost = currentCost;
				return;
			}

			// change pose tol for rotation and translation

			double predicted_x,predicted_y, predicted_z, predicted_roll, predicted_pitch, predicted_yaw;

			predicted_x= x - poseStep(0);
			predicted_y = y - poseStep(1);
			predicted_z = z - poseStep(2);
			predicted_roll = roll - poseStep(3);
			predicted_pitch = pitch - poseStep(4);
			predicted_yaw = yaw - poseStep(5);

			iterationTransform = Eigen::Affine3d::Identity();
			iterationTransform.translation() << predicted_x, predicted_y, predicted_z;
			iterationTransform.rotate (Eigen::AngleAxisd (predicted_roll, Eigen::Vector3d::UnitX()));
			iterationTransform.rotate (Eigen::AngleAxisd (predicted_pitch, Eigen::Vector3d::UnitY()));
			iterationTransform.rotate(Eigen::AngleAxisd (predicted_yaw, Eigen::Vector3d::UnitZ()));

			pcl::transformPointCloud(*scan2, *transformedScan, iterationTransform);
			double newCost = 0;

			computeCost(newCost, transformedScan);

			std::cout << "Cost after prediction: " << newCost << std::endl;

			double diff = currentCost - newCost;

			if (diff > 0) {

				if (fabs(diff) < tol) {
					// converged
					cout << "Change in cost is very minute. Ending iteration" << std::endl;
					resultantTransform = iterationTransform;
					cost = newCost;
					return;
				}

				std::cout << "Taking the step" << std::endl;

				lambda /= 10;

				x = predicted_x;
				y = predicted_y;
				z = predicted_z;
				roll = predicted_roll;
				pitch = predicted_pitch;
				yaw = predicted_yaw;

				std::cout << "x: " << x << std::endl;
				std::cout << "y: " << y << std::endl;
				std::cout << "z: " << z << std::endl;
				std::cout << "roll: " << roll << std::endl;
				std::cout << "pitch" << pitch << std::endl;
				std::cout << "yaw: " << yaw << std::endl;

				iterationComplete = true;

			} else {
				std::cout << "Increasing lambda and trying again" << std::endl;
				lambda *= 10;
			}

			// end lambda while
		}

		// end iteration while
	}

	resultantTransform = iterationTransform;
	cost = currentCost;
}

void svrOptimize::computeCost(double &cost, svr::PointCloudT::Ptr transformedScan) {

	cost = 0;
	svr::SVMap* svMap = opt_data.svMap;
	svr::SVMap::iterator svMapItr;
	SData::ScanIndexVector::iterator pItr;
	SData::ScanIndexVectorPtr indexVector;

	//
	// this iteration calcultes the cost
	for (svMapItr = svMap->begin(); svMapItr != svMap->end(); ++svMapItr) {

		SData::Ptr supervoxel = svMapItr->second;

		double d1 = supervoxel->getD1();
		double d2 = supervoxel->getD2();
		Eigen::Matrix3f covarianceInv = supervoxel->getCovarianceInverse();
		Eigen::Matrix3f covariance = supervoxel->getCovariance();
		Eigen::Vector4f mean = supervoxel->getCentroid();
		Eigen::Vector3f U;
		U << mean(0), mean(1), mean(2);

		indexVector = supervoxel->getScanBIndexVector();

		for (pItr = indexVector->begin(); pItr != indexVector->end(); ++pItr) {
			// calculate hessian, gradient contribution of individual points
			svr::PointT p = transformedScan->at(*pItr);
			Eigen::Vector3f X;
			X << p.x, p.y, p.z;

			float power = (X-U).transpose() * covarianceInv * (X-U);
			power = -d2 * power / 2;
			double exponentPower = exp(power);
			cost += d1 * exponentPower;
		}
	}

}

void svrOptimize::computeCostGradientHessian(double &cost, Eigen::VectorXf& g,
		Eigen::MatrixXf& H, svr::PointCloudT::Ptr transformedScan) {

	svr::SVMap* svMap = opt_data.svMap;
	svr::SVMap::iterator svMapItr;
	SData::ScanIndexVector::iterator pItr;
	SData::ScanIndexVectorPtr indexVector;

	// this iteration takes time
	for (svMapItr = svMap->begin(); svMapItr != svMap->end(); ++svMapItr) {

		SData::Ptr supervoxel = svMapItr->second;

		double d1 = supervoxel->getD1();
		double d2 = supervoxel->getD2();

		Eigen::Matrix3f covarianceInv = supervoxel->getCovarianceInverse();
		Eigen::Matrix3f covariance = supervoxel->getCovariance();
		Eigen::Vector4f mean = supervoxel->getCentroid();
		Eigen::Vector3f U;
		U << mean(0), mean(1), mean(2);

		indexVector = supervoxel->getScanBIndexVector();

		for (pItr = indexVector->begin(); pItr != indexVector->end(); ++pItr) {
			// calculate hessian, gradient contribution of individual points
			svr::PointT p = transformedScan->at(*pItr);
			Eigen::Vector3f X;
			X << p.x, p.y, p.z;

			// Using small angle approximation
			Eigen::MatrixXf Jacobian (3,6);
			Jacobian << 1,0,0,0,X(2),-X(1),
					0,1,0,-X(2),0,X(0),
					0,0,1,X(1),-X(0),0;

			float power = (X-U).transpose() * covarianceInv * (X-U);
			power = -d2 * power / 2;
			double exponentPower = exp(power);

			cost += d1 * exponentPower;

			for (int i=0; i<6; i++) {
				g(i) += (-d1 * d2 * exponentPower * (X-U).transpose() * covarianceInv * Jacobian.col(i));
			}

			for (int i = 0; i < 6; i++) {
				for (int j = i; j < 6; j++) {
					double r = -d1*d2*exponentPower;

					double r2 = -d2;
					r2 *= (X-U).transpose()*covarianceInv*Jacobian.col(i);
					r2 *= (X-U).transpose()*covarianceInv*Jacobian.col(j);

					double r3 = Jacobian.col(j).transpose()*covarianceInv*Jacobian.col(i);
					r = r * (r2+r3);
					H(i,j) = H(i,j) + r;
				}
			}

		}
	}
}


// Gauss Newton Optimization
void svrOptimize::optimizeUsingOriginalLMA(Eigen::Affine3d& resultantTransform, float& cost) {

	svr::PointCloudT::Ptr scan2 = opt_data.scan2;
	svr::SVMap* svMap = opt_data.svMap;
	Eigen::Affine3d last_transform = opt_data.t;

	double x,y,z,roll,pitch,yaw;
	svr_util::transform_get_translation_from_affine(last_transform, &x, &y, &z);
	svr_util::transform_get_rotation_from_affine(last_transform, &roll, &pitch, &yaw);
	std::cout << "Optimizing" << std::endl;
	std::cout << "x: " << x << std::endl;
	std::cout << "y: " << y << std::endl;
	std::cout << "z: " << z << std::endl;
	std::cout << "roll: " << roll << std::endl;
	std::cout << "pitch: " << pitch << std::endl;
	std::cout << "yaw: " << yaw << std::endl;

	bool debug = true;
	double tol = 1e-4, stepSize = 1., poseRotTol = 1e-15, poseTransTol = 1e-10;
	float lambda = 1e-3;

	svr::SVMap::iterator svMapItr;
	SData::ScanIndexVector::iterator pItr;
	SData::ScanIndexVectorPtr indexVector;
	Eigen::Affine3d iterationTransform;
	svr::PointCloudT::Ptr transformedScan =  boost::shared_ptr<svr::PointCloudT>(new svr::PointCloudT());
	double currentCost = 0;
	Eigen::MatrixXf H(6,6);
	Eigen::VectorXf g(6,1);

	iterationTransform = Eigen::Affine3d::Identity();
	iterationTransform.translation() << x,y,z;
	iterationTransform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
	iterationTransform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
	iterationTransform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

	// transform Scan
	pcl::transformPointCloud(*scan2, *transformedScan, iterationTransform);

	// Calculate Hessian and Gradient at last pose
	H.setZero();
	g.setZero();
	currentCost = 0;

	computeCostGradientHessian(currentCost, g, H, transformedScan);

	bool converged = true;
	for (int i = 0; i < 6; i++) {

		double gdiff = fabs((double)g(i));

		if (gdiff > tol) {
			converged = false;
			break;
		}
	}

	std::cout << "Current Cost: " << currentCost << std::endl;
	std::cout << "Gradient: " << std::endl << g << std::endl;

	if (converged) {
		resultantTransform = iterationTransform;
		cost = currentCost;
		return;
	}

	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			H(j,i) = H(i,j);
		}
	}

	std::cout << "Hessian: " << std::endl << H << std::endl;

	Eigen::MatrixXf A = H;


	bool iterationComplete = false;
	Eigen::VectorXf poseStep(6,1);

	while (!iterationComplete) {

		// till cost doesnt decrease
		// switch between gauss newton and gradient descent
		std::cout << "Lambda: " << lambda << std::endl;

		A = H;
		for (int i = 0; i < 6; i++) {
			A(i,i) *= (lambda + 1);
		}

		//			std::cout << "A: " << std::endl << A;
		//			std::cout << std::endl;
		poseStep = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g);
		poseStep = stepSize * poseStep;
		//		poseStep = stepSize * g;
		std::cout << "Pose Step: " << std::endl << poseStep << std::endl;

		//

		bool iterationProgressing = false;

		for (int k=0; k<6; k++) {
			double poseDiff = fabs((double)poseStep(k));
			if (k < 3 && poseDiff > poseTransTol) {
				iterationProgressing = true;
				break;
			} else if (k > 3 && poseDiff > poseRotTol) {
				iterationProgressing = true;
				break;
			}
		}

		if (!iterationProgressing) {
			// not converging
			std::cout << "Pose step too small" << std::endl;
			resultantTransform = iterationTransform;
			cost = currentCost;
			return;
		}

		// change pose tol for rotation and translation

		double predicted_x,predicted_y, predicted_z, predicted_roll, predicted_pitch, predicted_yaw;

		predicted_x= x - poseStep(0);
		predicted_y = y - poseStep(1);
		predicted_z = z - poseStep(2);
		predicted_roll = roll - poseStep(3);
		predicted_pitch = pitch - poseStep(4);
		predicted_yaw = yaw - poseStep(5);

		iterationTransform = Eigen::Affine3d::Identity();
		iterationTransform.translation() << predicted_x, predicted_y, predicted_z;
		iterationTransform.rotate (Eigen::AngleAxisd (predicted_roll, Eigen::Vector3d::UnitX()));
		iterationTransform.rotate (Eigen::AngleAxisd (predicted_pitch, Eigen::Vector3d::UnitY()));
		iterationTransform.rotate(Eigen::AngleAxisd (predicted_yaw, Eigen::Vector3d::UnitZ()));

		pcl::transformPointCloud(*scan2, *transformedScan, iterationTransform);
		double newCost = 0;

		computeCost(newCost, transformedScan);

		std::cout << "Cost after prediction: " << newCost << std::endl;

		double diff = currentCost - newCost;

		if (diff > 0) {

			if (fabs(diff) < tol) {
				// converged
				cout << "Change in cost is very minute. Ending iteration" << std::endl;
				resultantTransform = iterationTransform;
				cost = newCost;
				return;
			}

			std::cout << "Taking the step" << std::endl;

			lambda /= 10;

			x = predicted_x;
			y = predicted_y;
			z = predicted_z;
			roll = predicted_roll;
			pitch = predicted_pitch;
			yaw = predicted_yaw;

			std::cout << "x: " << x << std::endl;
			std::cout << "y: " << y << std::endl;
			std::cout << "z: " << z << std::endl;
			std::cout << "roll: " << roll << std::endl;
			std::cout << "pitch" << pitch << std::endl;
			std::cout << "yaw: " << yaw << std::endl;

			iterationComplete = true;

		} else {
			std::cout << "Increasing lambda and trying again" << std::endl;
			lambda *= 10;
		}

		// end lambda while
	}

	resultantTransform = iterationTransform;
	cost = currentCost;

}













