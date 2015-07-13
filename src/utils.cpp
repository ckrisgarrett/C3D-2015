/*
    File:   utils.cpp
    Author: Kris Garrett
    Date:   September 6, 2013
    
    Several functions are defined here as helper utilities to the rest of the 
    program.
*/


#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>

#ifdef USE_MPI
#include <mpi.h>
#endif


/*
    Implements abort for both serial and MPI code.
*/
extern "C"
void utils_abort()
{
    #ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    #else
    abort();
    #endif
}

double utils_norm1(int n, double *A)
{
    double oneNorm = 0.0;
    
    for(int i = 0; i < n; i++) {
        double oneNormTemp = 0;
        for(int j = 0; j < n; j++)
            oneNormTemp += fabs(A[i*n+j]);
        
        if(oneNormTemp > oneNorm)
            oneNorm = oneNormTemp;
    }
    
    return oneNorm;
}


void utils_getGaussianWeightsAndNodes(int n, double *w, double *mu)
{
    gsl_integration_glfixed_table *table = gsl_integration_glfixed_table_alloc(n);
    for(int i = 0; i < n; i++)
        gsl_integration_glfixed_point(-1, 1, i, &mu[i], &w[i], table);
}


double utils_getSphericalHarmonic(int m, int n, double mu, double phi)
{
    if(m < 0)
        return M_SQRT2 * sin(m * phi) * gsl_sf_legendre_sphPlm(n, -m, mu);
    else if(m > 0)
        return M_SQRT2 * cos(m * phi) * gsl_sf_legendre_sphPlm(n, m, mu);
    else
        return gsl_sf_legendre_sphPlm(n, 0, mu);
}


void dpotrs(const int n, const double * __restrict__ L, double * __restrict__ d)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            d[i] -= L[i+n*j] * d[j];
        }
        d[i] = d[i] / L[i*n+i];
    }
    
    for(int i = n-1; i >= 0; i--) {
        for(int j = n-1; j > i; j--) {
            d[i] -= L[j+n*i] * d[j];
        }
        d[i] = d[i] / L[i*n+i];
    }
}
