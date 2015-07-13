/*
 File:   cuda_solveH.cu
 Author: Kris Garrett
 Date:   April 29, 2013
*/

#include "cuda_headers.h"
#include "../global.h"
#include <stdio.h>
#include <cublas_v2.h>


#define CUDA_ERROR_SCRIPT  checkCudaError(__LINE__, __FILE__);
static
void checkCudaError(int line, char *filename)
{
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("[CUDA ERROR %d] %s: %d: %s\n", g_node, filename, line, cudaGetErrorString(error));
}


// The device variables.
// These are allocated on the CUDA cards in fobj_initialize.
static double *wDev;
static double *pDev;
static double *p2Dev;
static double *pTDev;
static double *alphasDev;
static double *hsDev;
static double *tempsDev;
static double *momentsDev;
static double *betaDev;
static double *betaTDev;
static double *betaValDev;
static int *betaColIndDev;
static int *betaRowPtrDev;

static cublasHandle_t handle;

/*
    Initialization.
*/
extern "C"
void solveHInit_cuda(int nm, int nm2, int nq, int batchSize, double *w, double *p, double *p2, 
                     double *beta, double *betaVal, int *betaColInd, int *betaRowPtr, int useGaunt)
{
    cublasStatus_t cublasStatus;
    double *pT;
    double *betaT;
    
    cublasStatus = cublasCreate(&handle);
    if(cublasStatus != CUBLAS_STATUS_SUCCESS)
        printf("cublasStatus != SUCCESS\n");
    
    cudaMalloc((void**)&wDev,       nq * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&pDev,       nq * nm * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&pTDev,      nq * nm * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&p2Dev,      nq * nm2 * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&alphasDev,  nm * batchSize * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&hsDev,      nm * nm * batchSize * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&tempsDev,   nq * batchSize * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&momentsDev, nm2 * batchSize * sizeof(double));  CUDA_ERROR_SCRIPT
    
    if(useGaunt != 0) {
        cudaMalloc((void**)&betaDev,    nm * nm * nm2 * sizeof(double));  CUDA_ERROR_SCRIPT
        cudaMalloc((void**)&betaTDev,    nm * nm * nm2 * sizeof(double));  CUDA_ERROR_SCRIPT
        cudaMalloc((void**)&betaValDev,    betaRowPtr[nm*nm] * sizeof(double));  CUDA_ERROR_SCRIPT
        cudaMalloc((void**)&betaColIndDev, betaRowPtr[nm*nm] * sizeof(int));  CUDA_ERROR_SCRIPT
        cudaMalloc((void**)&betaRowPtrDev, (nm * nm + 1) * sizeof(int));  CUDA_ERROR_SCRIPT
    }
    
    cudaMemcpy(wDev, w, nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemcpy(pDev, p, nm * nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemcpy(p2Dev, p2, nm2 * nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    
    if(useGaunt != 0) {
        cudaMemcpy(betaDev, beta, nm * nm * nm2 * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
        cudaMemcpy(betaValDev, betaVal, betaRowPtr[nm*nm] * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
        cudaMemcpy(betaColIndDev, betaColInd, betaRowPtr[nm*nm] * sizeof(int), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
        cudaMemcpy(betaRowPtrDev, betaRowPtr, (nm * nm + 1) * sizeof(int), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    }
    
    pT = new double[nm * nq];
    for(int q = 0; q < nq; q++) {
        for(int k = 0; k < nm; k++)
            pT[k * nq + q] = p[q * nm + k];
    }
    cudaMemcpy(pTDev, pT, nm * nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    delete[] pT;
    
    if(useGaunt != 0) {
        betaT = new double[nm * nm * nm2];
        for(int ij = 0; ij < nm * nm; ij++) {
            for(int k = 0; k < nm2; k++)
                betaT[k * nm * nm + ij] = beta[ij * nm2 + k];
        }
        cudaMemcpy(betaTDev, betaT, nm * nm * nm2 * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
        delete[] betaT;
    }
}


/*
    Fills in the array e^(alpha^T p).
*/
__global__
void kernel_fillTempArray(int nm, int nq, int batchSize, 
                          const double * __restrict alphas, const double * __restrict w, 
                          const double * __restrict pT, double * __restrict tempsArray)
{
    int batchIndex;
    int index;
    double temp;
    
    batchIndex = blockIdx.x;
    for(int q = threadIdx.x; q < nq; q += blockDim.x) {
        index = batchIndex * nq + q;
        
        temp = 0.0;
        for(int k = 0; k < nm; k++) {
            temp += alphas[batchIndex * nm + k] * pT[k * nq + q];
        }
        tempsArray[index] = exp(temp) * w[q];
    }
}


__global__
void kernel_setH(int nm, int nq, int batchSize, 
                 const double * __restrict tempsArray, const double * __restrict p, 
                 double * __restrict hs)
{
    int batchIndex = blockIdx.x;
    //int i = threadIdx.y;
    //int j = threadIdx.x;
    
    
    for(int i = threadIdx.y; i < nm; i += 16) {
    for(int j = threadIdx.x; j < nm; j += 16) {
        int index = batchIndex * nm * nm + i * nm + j;
        if(batchIndex < batchSize) {
            hs[index] = 0;
            
            // Do the integral part of the calculation.
            for(int q = 0; q < nq; q++) {
                hs[index] += tempsArray[batchIndex * nq + q] * p[q * nm + i] * p[q * nm + j];
            }
        }
    }}
}


__global__
void kernel_setMoments(int nm2, int nq, int batchSize, const double * __restrict tempsArray, 
                       const double * __restrict p2, double * __restrict moments)
{
    int batch = blockIdx.x;
    
    for(int k = threadIdx.x; k < nm2; k += blockDim.x) {
        moments[batch * nm2 + k] = 0.0;
        
        for(int q = 0; q < nq; q++) {
            moments[batch * nm2 + k] += tempsArray[batch * nq + q] * p2[q * nm2 + k];
        }
    }
}


__global__
void kernel_setHCG(int nm, int nm2, int batchSize, 
                   const double * __restrict beta, const double * __restrict moments, 
                   double * __restrict hs)
{
    int batch = blockIdx.x;
    
    for(int index = threadIdx.x; index < nm * nm; index += blockDim.x) {
        hs[batch * nm * nm + index] = 0;
        
        for(int k = 0; k < nm2; k++) {
            //hs[batch * nm * nm + index] += beta[index * nm2 + k] * moments[batch * nm2 + k];
            hs[batch * nm * nm + index] += beta[k * nm * nm + index] * moments[batch * nm2 + k];
        }
    }
}


__global__
void kernel_setHGauntSparse(int nm, int nm2, int batchSize, 
                   const int * __restrict betaRowPtr, const int * __restrict betaColInd, 
                   const double * __restrict betaVal, const double * __restrict moments, 
                   double * __restrict hs)
{
    int batch = blockIdx.x;
    
    for(int index = threadIdx.x; index < nm * nm; index += blockDim.x) {
        hs[batch * nm * nm + index] = 0;
        
        for(int k = betaRowPtr[index]; k < betaRowPtr[index+1]; k++) {
            hs[batch * nm * nm + index] += moments[batch * nm2 + betaColInd[k]] * betaVal[k];
        }
    }
}


/*
    Solve for a batch of Hessians using CUDA.
*/
extern "C"
void solveH_cuda(int nm, int nm2, int nq, int batchSize, double *alphas, double *hs, 
                 int useGaunt, int useGauntSparse)
{
    int nt1 = 128;
    int nb1 = batchSize;
    dim3 nt2(16,16);
    int nb2 = batchSize;
    int nt3 = 128;
    int nb3 = batchSize;
    
    
    cudaMemcpy(alphasDev, alphas, nm * batchSize * sizeof(double), 
               cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    
    kernel_fillTempArray<<<nb1, nt1>>>(nm, nq, batchSize, alphasDev, wDev, pTDev, tempsDev);
    if(useGaunt == 0) {
        kernel_setH<<<nb2, nt2>>>(nm, nq, batchSize, tempsDev, pDev, hsDev);
    }
    else {
        //kernel_setMoments<<<nb3,nt3>>>(nm2, nq, batchSize, tempsDev, p2Dev, momentsDev);
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nm2, batchSize, nq, 
                    &alpha, p2Dev, nm2, tempsDev, nq, &beta, momentsDev, nm2);
        
        if(useGauntSparse == 0) {
            //kernel_setHCG<<<nb3, nt3>>>(nm, nm2, batchSize, betaDev, momentsDev, hsDev);
            kernel_setHCG<<<nb3, nt3>>>(nm, nm2, batchSize, betaTDev, momentsDev, hsDev);
        }
        else {
            kernel_setHGauntSparse<<<nb3, nt3>>>(nm, nm2, batchSize, 
                            betaRowPtrDev, betaColIndDev, betaValDev, momentsDev, hsDev);
        }
    }
    
    cudaMemcpy(hs, hsDev, nm * nm * batchSize * sizeof(double), 
               cudaMemcpyDeviceToHost);  CUDA_ERROR_SCRIPT
}

