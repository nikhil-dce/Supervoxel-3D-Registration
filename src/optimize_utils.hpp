#ifndef DGC_TRANSFORM_H
#define DGC_TRANSFORM_H

#include <vector>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit_nlin.h>

/*
 *
 * These functions have been taken from Segal GICP git code
 * Reference: https://github.com/avsegal/gicp
 *
 */

double mat_inner_prod(gsl_matrix const* mat1, gsl_matrix const* mat2) {
	double r = 0.;
	int n = mat1->size1;

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) { // tr(mat1^t.mat2)
			r += gsl_matrix_get(mat1, j, i)*gsl_matrix_get(mat2, i, j);
		}
	}

	return r;
}

/*
 *
 * dR / dPhi
 * dR / dTheta
 * dR / dPsi
 */

void compute_dr(gsl_vector const* x, gsl_matrix const* gsl_temp_mat_r, gsl_vector *g) {
	double dR_dPhi[3][3];
	double dR_dTheta[3][3];
	double dR_dPsi[3][3];
	gsl_matrix_view gsl_d_rx = gsl_matrix_view_array(&dR_dPhi[0][0],3, 3);
	gsl_matrix_view gsl_d_ry = gsl_matrix_view_array(&dR_dTheta[0][0],3, 3);
	gsl_matrix_view gsl_d_rz = gsl_matrix_view_array(&dR_dPsi[0][0],3, 3);

	double phi = gsl_vector_get(x ,3);
	double theta = gsl_vector_get(x ,4);
	double psi = gsl_vector_get(x ,5);

	double cphi = cos(phi), sphi = sin(phi);
	double ctheta = cos(theta), stheta = sin(theta);
	double cpsi = cos(psi), spsi = sin(psi);

	dR_dPhi[0][0] = 0.;
	dR_dPhi[1][0] = 0.;
	dR_dPhi[2][0] = 0.;

	dR_dPhi[0][1] = sphi*spsi + cphi*cpsi*stheta;
	dR_dPhi[1][1] = -cpsi*sphi + cphi*spsi*stheta;
	dR_dPhi[2][1] = cphi*ctheta;

	dR_dPhi[0][2] = cphi*spsi - cpsi*sphi*stheta;
	dR_dPhi[1][2] = -cphi*cpsi - sphi*spsi*stheta;
	dR_dPhi[2][2] = -ctheta*sphi;

	dR_dTheta[0][0] = -cpsi*stheta;
	dR_dTheta[1][0] = -spsi*stheta;
	dR_dTheta[2][0] = -ctheta;

	dR_dTheta[0][1] = cpsi*ctheta*sphi;
	dR_dTheta[1][1] = ctheta*sphi*spsi;
	dR_dTheta[2][1] = -sphi*stheta;

	dR_dTheta[0][2] = cphi*cpsi*ctheta;
	dR_dTheta[1][2] = cphi*ctheta*spsi;
	dR_dTheta[2][2] = -cphi*stheta;

	dR_dPsi[0][0] = -ctheta*spsi;
	dR_dPsi[1][0] = cpsi*ctheta;
	dR_dPsi[2][0] = 0.;

	dR_dPsi[0][1] = -cphi*cpsi - sphi*spsi*stheta;
	dR_dPsi[1][1] = -cphi*spsi + cpsi*sphi*stheta;
	dR_dPsi[2][1] = 0.;

	dR_dPsi[0][2] = cpsi*sphi - cphi*spsi*stheta;
	dR_dPsi[1][2] = sphi*spsi + cphi*cpsi*stheta;
	dR_dPsi[2][2] = 0.;

	// set d/d_rx = tr(dR_dPhi'*gsl_temp_mat_r) [= <dR_dPhi, gsl_temp_mat_r>]
	gsl_vector_set(g, 3, mat_inner_prod(&gsl_d_rx.matrix, gsl_temp_mat_r));
	// set d/d_ry = tr(dR_dTheta'*gsl_temp_mat_r) = [<dR_dTheta, gsl_temp_mat_r>]
	gsl_vector_set(g, 4, mat_inner_prod(&gsl_d_ry.matrix, gsl_temp_mat_r));
	// set d/d_rz = tr(dR_dPsi'*gsl_temp_mat_r) = [<dR_dPsi, gsl_temp_mat_r>]
	gsl_vector_set(g, 5, mat_inner_prod(&gsl_d_rz.matrix, gsl_temp_mat_r));

}

#endif
