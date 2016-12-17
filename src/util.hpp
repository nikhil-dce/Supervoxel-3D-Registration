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

}
