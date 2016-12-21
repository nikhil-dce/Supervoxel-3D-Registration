
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

#define N_DIMEN 6

namespace svr_optimize {
    
    struct svr_opti_data {
        
        svr::SVMap* svMap;
        svr::PointCloudT::Ptr scan1;
        svr::PointCloudT::Ptr scan2;
        
    };
    
    Eigen::Affine3d
    optimize(svr_optimize_data opt_data) {
        
        const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
        gsl_multimin_fdfminimizer *gsl_minimizer = NULL;
        
        gsl_vector *baseX;
        
        bool debug = true;
        int max_iter = 20;
        double line_search_tol = .01;
        double gradient_tol = 1e-2;
        double step_size = 1.;
        
        // set up the gsl function_fdf struct
        gsl_multimin_function_fdf func;
        func.f = f;
        func.df = df;
        func.fdf = fdf;
        func.n = N_DIMEN;
        func.params = &opt_data;
        
        size_t iter = 0;
        int status;
        double size;
        
        baseX = gsl_vector_alloc (6);
        gsl_vector_set (baseX, 0, 0);
        gsl_vector_set (baseX, 1, 0);
        gsl_vector_set (baseX, 2, 0);
        gsl_vector_set (baseX, 3, 0);
        gsl_vector_set (baseX, 4, 0);
        gsl_vector_set (baseX, 5, 0);
        
        gsl_minimizer = gsl_multimin_fdfminimizer_alloc (T, N_DIMEN);
        gsl_multimin_fdfminimizer_set (gsl_minimizer, &minex_func, baseX, step_size, line_search_tol);
        
        while (status == GSL_CONTINUE && iter < max_iter) {
            
            iter++;
            status = gsl_multimin_fdfminimizer_iterate(gsl_minimzer);
            
            if(debug)
                cout << iter << "\t\t" << gsl_minimizer->f << "\t\t" << Status() <<endl;
            
            if (status)
                break;
            
            status = gsl_multimin_test_gradient (gsl_minimzer->gradient, gradient_tol);
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
            gsl_multimin_fminimizer_free(gsl_minimzer);
            
            return resultantTransform;
        }
        
        std::cout << "Algorithm failed to converge" << std::endl;
        gsl_vector_free(baseX);
        gsl_multimin_fminimizer_free(gsl_minimzer);
        
        return Eigen::Affine3d::Zero();
        
    }
    
}


#endif /* svr_optimize_h */
