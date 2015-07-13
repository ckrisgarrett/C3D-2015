/*
    File:   flux_kinetic.cpp
    Author: Kris Garrett
    Date:   September 16, 2013
*/

#include "utils.h"
#include "global.h"
#include "cuda/cuda_headers.h"
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


/* 
    Given alpha and the quadrature point, calculate the ansatz at that quadrature point.
*/
static 
double computeAnsatzPn(double *alpha, int q)
{
    double kinetic = 0.0;
    for(int m = 0; m < g_numMoments; m++)
        kinetic += alpha[m] * g_m[IM(q,m)];
    
    return kinetic;
}
static 
double computeAnsatzMn(double *alpha, int q)
{
    double kinetic = 0.0;
    for(int m = 0; m < g_numMoments; m++)
        kinetic += alpha[m] * g_m[IM(q,m)];
    
    return exp(kinetic);
}


/*
    Returns 0 if xy < 1
    Returns min(x,y) if x > 0 and y > 0
    Returns -min(|x|,|y|) if x < 0 and y < 0
*/
static
double minmod(double x, double y)
{
    return SIGN(1.0,x)* MAX(0.0, MIN(fabs(x),y*SIGN(1.0,x) ) );
}


/*
    Computes the undivided differences with limiter.
*/
static
double slopefit(double left, double center, double right)
{
    return minmod(g_theta*(right-center),
                  minmod(0.5*(right-left), g_theta*(center-left)) );
}


/*
    Solves for the flux at each grid cell.
*/
void solveFluxKinetic(double *u, double *flux, double *alpha, double dx, double dy, double dz)
{
    double k1, k2, k3, k4, fluxX, fluxY, fluxZ;
    static bool firstTime = true;
    static double *ansatzGrid;
    #ifdef USE_CUDA
    int isPn;
    #endif
    
    // First time initialization.
    if(firstTime) {
        firstTime = false;
        ansatzGrid = (double*)malloc((g_gX[3] - g_gX[0] + 1) * 
            (g_gY[3] - g_gY[0] + 1) * (g_gZ[3] - g_gZ[0] + 1) * sizeof(double));
    }
    
    
    // Try to use the cuda solver.
    #ifdef USE_CUDA
    if(g_cudaKinetic) {
        isPn = (g_solver == SOLVER_PN) ? 1 : 0;
        solveFluxKinetic_cuda((g_gX[3] - g_gX[0] + 1), (g_gY[3] - g_gY[0] + 1), 
                              (g_gZ[3] - g_gZ[0] + 1), g_numMoments, g_numQuadPoints, 
                              dx, dy, dz, alpha, flux, g_xi, g_eta, g_mu, g_w, g_m, isPn);
        return;
    }
    #endif
    
    
    // Use the CPU solver.
    // Zero flux.
    #pragma omp parallel for
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
    for(int m = 0; m < g_numMoments; m++) {
        flux[IU(i,j,k,m)] = 0.0;
    }}}}

    
    // Add up each quadrature's flux.
    for(int q = 0; q < g_numQuadPoints; q++) {
        
        if(g_solver == SOLVER_PN) {
            #pragma omp parallel for
            for(int i = g_gX[0]; i <= g_gX[3]; i++) {
            for(int j = g_gY[0]; j <= g_gY[3]; j++) {
            for(int k = g_gZ[0]; k <= g_gZ[3]; k++) {
                ansatzGrid[I3D(i,j,k)] = computeAnsatzPn(&alpha[IU(i,j,k,0)], q);
            }}}
        }
        else {
            #pragma omp parallel for
            for(int i = g_gX[0]; i <= g_gX[3]; i++) {
            for(int j = g_gY[0]; j <= g_gY[3]; j++) {
            for(int k = g_gZ[0]; k <= g_gZ[3]; k++) {
                ansatzGrid[I3D(i,j,k)] = computeAnsatzMn(&alpha[IU(i,j,k,0)], q);
            }}}
        }
        
        #pragma omp parallel for private(k1,k2,k3,k4,fluxX,fluxY,fluxZ)
        for(int i = g_gX[1]; i <= g_gX[2]; i++) {
        for(int j = g_gY[1]; j <= g_gY[2]; j++) {
        for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
            if(g_xi[q] > 0) {
                k1 = ansatzGrid[I3D(i-2,j,k)];
                k2 = ansatzGrid[I3D(i-1,j,k)];
                k3 = ansatzGrid[I3D(i,j,k)];
                k4 = ansatzGrid[I3D(i+1,j,k)];
                
                fluxX = k3 + 0.5 * slopefit(k2, k3, k4) - 
                    k2 - 0.5 * slopefit(k1, k2, k3);
                fluxX = fluxX * g_xi[q] * g_w[q];
            }
            else {
                k1 = ansatzGrid[I3D(i-1,j,k)];
                k2 = ansatzGrid[I3D(i,j,k)];
                k3 = ansatzGrid[I3D(i+1,j,k)];
                k4 = ansatzGrid[I3D(i+2,j,k)];
                
                fluxX = k3 - 0.5 * slopefit(k2, k3, k4) - 
                    k2 + 0.5 * slopefit(k1, k2, k3);
                fluxX = fluxX * g_xi[q] * g_w[q];
            }
            if(g_eta[q] > 0) {
                k1 = ansatzGrid[I3D(i,j-2,k)];
                k2 = ansatzGrid[I3D(i,j-1,k)];
                k3 = ansatzGrid[I3D(i,j,k)];
                k4 = ansatzGrid[I3D(i,j+1,k)];
                
                fluxY = k3 + 0.5 * slopefit(k2, k3, k4) - 
                    k2 - 0.5 * slopefit(k1, k2, k3);
                fluxY = fluxY * g_eta[q] * g_w[q];
            }
            else {
                k1 = ansatzGrid[I3D(i,j-1,k)];
                k2 = ansatzGrid[I3D(i,j,k)];
                k3 = ansatzGrid[I3D(i,j+1,k)];
                k4 = ansatzGrid[I3D(i,j+2,k)];
                
                fluxY = k3 - 0.5 * slopefit(k2, k3, k4) - 
                    k2 + 0.5 * slopefit(k1, k2, k3);
                fluxY = fluxY * g_eta[q] * g_w[q];
            }
            if(g_mu[q] > 0) {
                k1 = ansatzGrid[I3D(i,j,k-2)];
                k2 = ansatzGrid[I3D(i,j,k-1)];
                k3 = ansatzGrid[I3D(i,j,k)];
                k4 = ansatzGrid[I3D(i,j,k+1)];
                
                fluxZ = k3 + 0.5 * slopefit(k2, k3, k4) - 
                    k2 - 0.5 * slopefit(k1, k2, k3);
                fluxZ = fluxZ * g_mu[q] * g_w[q];
            }
            else {
                k1 = ansatzGrid[I3D(i,j,k-1)];
                k2 = ansatzGrid[I3D(i,j,k)];
                k3 = ansatzGrid[I3D(i,j,k+1)];
                k4 = ansatzGrid[I3D(i,j,k+2)];
                
                fluxZ = k3 - 0.5 * slopefit(k2, k3, k4) - 
                    k2 + 0.5 * slopefit(k1, k2, k3);
                fluxZ = fluxZ * g_mu[q] * g_w[q];
            }
            
            
            for(int m = 0; m < g_numMoments; m++) {
                flux[IU(i,j,k,m)] += g_m[IM(q,m)] * 
                    (fluxX / dx + fluxY / dy + fluxZ / dz);
            }
        }}}
    }
}

