/*
    File:   cuda_headers.h
    Author: Kris Garrett
    Date:   September 17, 2013
*/

#ifndef __CUDA_HEADERS_H
#define __CUDA_HEADERS_H

extern "C"
void solveFluxKinetic_cuda(int nx, int ny, int nz, int nm, int nq, double dx, double dy, 
                           double dz, double *alpha, double *flux, double *xi, 
                           double *eta, double *mu, double *w, double *p, int isPn);

extern "C"
void solveH_cuda(int nm, int nm2, int nq, int batchSize, double *alphas, double *hs, 
                 int useGaunt, int useGauntSparse);

extern "C"
void solveHInit_cuda(int nm, int nm2, int nq, int batchSize, double *w, double *p, double *p2, 
                     double *beta, double *betaVal, int *betaColInd, int *betaRowPtr, int useGaunt);

#endif
