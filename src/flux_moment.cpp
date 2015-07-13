/*
    File:   flux_moment.cpp
    Author: Kris Garrett
    Date:   September 16, 2013
*/

#include "global.h"
#include <stdlib.h>


/*
    Calculates a matrix-vector product.
*/
static 
void mvProduct(int n, double beta, double *A, double *x, double *b)
{
    for(int i = 0; i < n; i++) 
        b[i] = beta * b[i];
    
    for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
        b[i] += A[i*n+j] * x[j];
    }}
}


/*
    Scales flux on the entire grid.
*/
void solveFluxMoment(double *u, double *flux, double dx, double dy, double dz)
{
    static bool firstTime = true;
    static double *temp;
    static double *xFlux;
    static double *yFlux;
    static double *zFlux;
    #pragma omp threadprivate(temp,xFlux,yFlux,zFlux)
    
    
    // First time initialization.
    if(firstTime) {
        firstTime = false;
        
        #pragma omp parallel
        {
            temp  = (double*)malloc(g_numMoments * sizeof(double));
            xFlux  = (double*)malloc(g_numMoments * sizeof(double));
            yFlux  = (double*)malloc(g_numMoments * sizeof(double));
            zFlux  = (double*)malloc(g_numMoments * sizeof(double));
        }
    }
    
    
    #pragma omp parallel for
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        // x Flux
        for(int m = 0; m < g_numMoments; m++) {
            temp[m] = 0.25*u[IU(i+1,j,k,m)] + 0.75*u[IU(i,j,k,m)] - 
                1.25*u[IU(i-1,j,k,m)] + 0.25*u[IU(i-2,j,k,m)];
        }
        mvProduct(g_numMoments, 0.0, g_pnFluxOperatorXiPlus, temp, xFlux);
        
        for(int m = 0; m < g_numMoments; m++) {
            temp[m] = -0.25*u[IU(i+2,j,k,m)] + 1.25*u[IU(i+1,j,k,m)] - 
                0.75*u[IU(i,j,k,m)] - 0.25*u[IU(i-1,j,k,m)];
        }
        mvProduct(g_numMoments, 1.0, g_pnFluxOperatorXiMinus, temp, xFlux);
        
        
        // y Flux
        for(int m = 0; m < g_numMoments; m++) {
            temp[m] = 0.25*u[IU(i,j+1,k,m)] + 0.75*u[IU(i,j,k,m)] - 
                1.25*u[IU(i,j-1,k,m)] + 0.25*u[IU(i,j-2,k,m)];
        }
        mvProduct(g_numMoments, 0.0, g_pnFluxOperatorEtaPlus, temp, yFlux);
        
        for(int m = 0; m < g_numMoments; m++) {
            temp[m] = -0.25*u[IU(i,j+2,k,m)] + 1.25*u[IU(i,j+1,k,m)] - 
                0.75*u[IU(i,j,k,m)] - 0.25*u[IU(i,j-1,k,m)];
        }
        mvProduct(g_numMoments, 1.0, g_pnFluxOperatorEtaMinus, temp, yFlux);
        
        
        // z Flux
        for(int m = 0; m < g_numMoments; m++) {
            temp[m] = 0.25*u[IU(i,j,k+1,m)] + 0.75*u[IU(i,j,k,m)] - 
                1.25*u[IU(i,j,k-1,m)] + 0.25*u[IU(i,j,k-2,m)];
        }
        mvProduct(g_numMoments, 0.0, g_pnFluxOperatorMuPlus, temp, zFlux);
        
        for(int m = 0; m < g_numMoments; m++) {
            temp[m] = -0.25*u[IU(i,j,k+2,m)] + 1.25*u[IU(i,j,k+1,m)] - 
                0.75*u[IU(i,j,k,m)] - 0.25*u[IU(i,j,k-1,m)];
        }
        mvProduct(g_numMoments, 1.0, g_pnFluxOperatorMuMinus, temp, zFlux);
        
        
        // Add fluxes together.
        for(int m = 0; m < g_numMoments; m++) 
            flux[IU(i,j,k,m)] = xFlux[m] / dx + yFlux[m] / dy + zFlux[m] / dz;
    }}}
}
