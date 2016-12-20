#define NORMAL_3D_CONSTANT 0.063493636
#define INTEGRAL_STEP 0.1

namespace svr_util {

template <typename Type>
inline Type square(Type x)
{
	return x * x; // will only work on types for which there is the * operator.
}

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
    
double calculateNormalProbabilityForPoint(float x, float y, float z, Eigen::Matrix3f covariance, Eigen::Vector4f mean) {
    
    std::cout << "Covariance: " << std::endl;
    std::cout << covariance;
    std::cout << std::endl;
    
    std::cout << "Mean: " << std::endl;
    std::cout << mean;
    std::cout << std::endl;
    
    std::cout << x << ' ' << y << ' ' << z << std::endl;
    
    double probability;
    
    Eigen::Vector3f X;
    X << x, y, z;
    
    Eigen::Vector3f U;
    U << mean(0), mean(1), mean(2);
    
    float power = (X-U).transpose() * covariance.inverse() * (X-U);
    probability = NORMAL_CONSTANT_3K * exp(-1 * power / 2) / covariance.determinant();
    
    std::cout << probability << std::endl;
    
    return probability;
}

double calculateApproximateIntegralForVoxel(float ax, float bx, float ay, float by, float az, float bz, Eigen::Matrix3f covariance, Eigen::Vector4f mean) {

    double integral = 0;
    float delta = INTEGRAL_STEP;

    // bx - ax = by - ay = bz - az => Voxel Resolution
    
    int m = fabs ( (bx-ax) / delta);

    
    std::vector<float> weights;
    
    delta /= 3;
    
    // calcute simpsons weights
    weights.push_back(delta);
    
    for (int i == 1; i <= m-2; i++) {
        if (i%2 == 0)
            weights.push_back(2 * delta);
        else
            weights.push_back(4 * delta);
    }
    
    weights.push_back(delta);
    
    int counterX(0), counterY(0), counterZ(0);
    float x, y, z;
    float wx, wy, wz;
    
    for (counterX = 0; counterX < m; counterX ++) {
        
        x = ax + (counterX * delta);
        wx = weights[counterX];
        
        for (counterY = 0; counterY < m; counterY ++) {
            
            y = ay + (counterY * delta);
            wy = weights[counterY];
            
            for (counterZ = 0; counterZ < m; counterZ ++) {
            
                z = az + (counterZ * delta);
                wz = weights[counterZ];
                
                float weight = wx * wy * wz;
                double pro = calculateNormalProbabilityForPoint(x, y, z, covariance, mean);
                integral += pro * weight;
            
            }
        }
    }
    
}
    
}









