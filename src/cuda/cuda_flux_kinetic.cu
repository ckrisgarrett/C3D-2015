/*
    File:   solve_flux_cuda.cu
    Author: Kris Garrett
    Date:   April 19, 2013
*/

#include "cuda_headers.h"
#include "../utils.h"
#include "../global.h"
#include <cublas_v2.h>


#define CUDA_ERROR_SCRIPT  checkCudaError(__LINE__, __FILE__);
static
void checkCudaError(int line, char *filename)
{
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("[CUDA ERROR %d] %s: %d: %s\n", g_node, filename, line, cudaGetErrorString(error));
}


// GPU device variables
static double *alphaDev;
static double *pDev;
static double *ansatzDev;
static double *wDev;
static double *xiDev;
static double *etaDev;
static double *muDev;
static double *fluxDev;
static double *fluxTempDev;

static cublasHandle_t handle;


/*
    Initialization for solveFlux_cuda.
*/
static 
void cudaInit(int nx, int ny, int nz, int nm, int nq, double *w, double *p, 
              double *xi, double *eta, double *mu)
{
    cublasStatus_t cublasStatus;
    
    
    cublasStatus = cublasCreate(&handle);
    if(cublasStatus != CUBLAS_STATUS_SUCCESS)
        printf("cublasStatus != SUCCESS\n");
    
    
    cudaMalloc((void**)&alphaDev,    nx * ny * nz * nm * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&pDev,        nq * nm * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&ansatzDev,   nx * ny * nz * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&wDev,        nq * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&xiDev,       nq * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&etaDev,      nq * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&muDev,       nq * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&fluxDev,     nx * ny * nz * nm * sizeof(double));  CUDA_ERROR_SCRIPT
    cudaMalloc((void**)&fluxTempDev, nx * ny * nz * sizeof(double));  CUDA_ERROR_SCRIPT
    
    cudaMemcpy(pDev,   p,   nq * nm * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemcpy(wDev,   w,   nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemcpy(xiDev,  xi,  nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemcpy(etaDev, eta, nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemcpy(muDev,  mu,  nq * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
}


/*
    Helper functions to calculate slopes.
*/
__device__
double minmod(double x, double y)
{
    return SIGN(1.0,x) * MAX(0.0, MIN(fabs(x), y * SIGN(1.0, x)));
}

__device__
double slopefit(double left, double center, double right)
{
    const double theta = 2.0;
    return minmod(theta*(right-center), minmod(0.5*(right-left), theta*(center-left)));
}


/*
    Computes the ansatz at quadrature index q for the entire grid.
*/
__global__
void kernel_computeAnsatz(int n, int nm, int q, double *alpha, double *p, double *ansatz, int isPn)
{
    __shared__ double temp[16][16];
    int space_index, batch_index;
    int k;
    double ans;
    
    
    temp[threadIdx.y][threadIdx.x] = 0.0;
    __syncthreads();
    
    
    space_index = blockIdx.x * blockDim.x + threadIdx.y;
    if(space_index < n) {
        for(batch_index = 0; batch_index < nm; batch_index += 16) {
            k = batch_index + threadIdx.x;
            
            if(k < nm)
                temp[threadIdx.y][threadIdx.x] += alpha[space_index * nm + k] * p[q*nm+k];
        }
        __syncthreads();
        
        if(threadIdx.y == 0) {
            ans = 0.0;
            for(int i = 0; i < 16; i++)
                ans += temp[threadIdx.x][i];
            
            if(isPn == 0)
                ans = exp(ans);
            
            ansatz[blockIdx.x * blockDim.x + threadIdx.x] = ans;
        }
    }
}


/*
    Begins flux calculation.
*/
__global__
void kernel_solveFluxTemp(int nx, int ny, int nz, int q, 
                          double dx, double dy, double dz, 
                          double *xi, double *eta, double *mu, 
                          double *w, double *ansatz, double *fluxTemp)
{
    int i, j, k, space_index;
    double flux1, flux2, flux3;
    double xi_q, eta_q, mu_q, w_q;
    double a1, a2, a3, a4;
    
    
    xi_q = xi[q];
    eta_q = eta[q];
    mu_q = mu[q];
    w_q = w[q];
    flux1 = 0.0;
    flux2 = 0.0;
    flux3 = 0.0;
    
    k = threadIdx.x + 2;
    j = blockIdx.x + 2;
    for(i = 2; i < nx - 2; i++) {
        if(xi_q > 0) {
            a1 = ansatz[k + nz * (j + ny * (i-2))];
            a2 = ansatz[k + nz * (j + ny * (i-1))];
            a3 = ansatz[k + nz * (j + ny * (i  ))];
            a4 = ansatz[k + nz * (j + ny * (i+1))];
            
            flux1 = a3 + 0.5 * slopefit(a2, a3, a4) - 
                a2 - 0.5 * slopefit(a1, a2, a3);
            flux1 = flux1 * xi_q * w_q;
        }
        else {
            a1 = ansatz[k + nz * (j + ny * (i-1))];
            a2 = ansatz[k + nz * (j + ny * (i  ))];
            a3 = ansatz[k + nz * (j + ny * (i+1))];
            a4 = ansatz[k + nz * (j + ny * (i+2))];
            
            flux1 = a3 - 0.5 * slopefit(a2, a3, a4) - 
                a2 + 0.5 * slopefit(a1, a2, a3);
            flux1 = flux1 * xi_q * w_q;
        }
        if(eta_q > 0) {
            a1 = ansatz[k + nz * ((j-2) + ny * i)];
            a2 = ansatz[k + nz * ((j-1) + ny * i)];
            a3 = ansatz[k + nz * ((j  ) + ny * i)];
            a4 = ansatz[k + nz * ((j+1) + ny * i)];
            
            flux2 = a3 + 0.5 * slopefit(a2, a3, a4) - 
                a2 - 0.5 * slopefit(a1, a2, a3);
            flux2 = flux2 * eta_q * w_q;
        }
        else {
            a1 = ansatz[k + nz * ((j-1) + ny * i)];
            a2 = ansatz[k + nz * ((j  ) + ny * i)];
            a3 = ansatz[k + nz * ((j+1) + ny * i)];
            a4 = ansatz[k + nz * ((j+2) + ny * i)];
            
            flux2 = a3 - 0.5 * slopefit(a2, a3, a4) - 
                a2 + 0.5 * slopefit(a1, a2, a3);
            flux2 = flux2 * eta_q * w_q;
        }
        if(mu_q > 0) {
            a1 = ansatz[(k-2) + nz * (j + ny * i)];
            a2 = ansatz[(k-1) + nz * (j + ny * i)];
            a3 = ansatz[(k  ) + nz * (j + ny * i)];
            a4 = ansatz[(k+1) + nz * (j + ny * i)];
            
            flux3 = a3 + 0.5 * slopefit(a2, a3, a4) - 
                a2 - 0.5 * slopefit(a1, a2, a3);
            flux3 = flux3 * mu_q * w_q;
        }
        else {
            a1 = ansatz[(k-1) + nz * (j + ny * i)];
            a2 = ansatz[(k  ) + nz * (j + ny * i)];
            a3 = ansatz[(k+1) + nz * (j + ny * i)];
            a4 = ansatz[(k+2) + nz * (j + ny * i)];
            
            flux3 = a3 - 0.5 * slopefit(a2, a3, a4) - 
                a2 + 0.5 * slopefit(a1, a2, a3);
            flux3 = flux3 * mu_q * w_q;
        }
        
        space_index = k + nz * (j + ny * i);
        fluxTemp[space_index] = flux1 / dx + flux2 / dy + flux3 / dz;
    }
}


/*
    Finishes flux calculation.
*/
__global__
void kernel_solveFlux(int nm, int q, double *fluxTemp, double *p, double *flux)
{
    __shared__ double pShared[256];
    int space_index, batch_index;
    int k;
    
    k = threadIdx.x + threadIdx.y * 16;
    if(k < nm)
        pShared[k] = p[q * nm + k];
    __syncthreads();
    
    space_index = blockIdx.x * blockDim.x + threadIdx.y;
    
    for(batch_index = 0; batch_index < nm; batch_index += 16) {
        k = threadIdx.x + batch_index;
        if(k < nm)
            //flux[space_index * nm + m] += p[q * nm + m] * fluxTemp[space_index];
            flux[space_index * nm + k] += pShared[k] * fluxTemp[space_index];
    }
}


/*
    Solve flux on GPU driver.

    nx:    size of grid in x-direction
    ny:    size of grid in y-direction
    nm:    number of moments
    q:     quadrature index
    theta: value for slopefit
    alpha: grid of alpha vectors
    flux:  grid of fluxes
*/
extern "C"
void solveFluxKinetic_cuda(int nx, int ny, int nz, int nm, int nq, double dx, double dy, 
                           double dz, double *alpha, double *flux, double *xi, 
                           double *eta, double *mu, double *w, double *p, int isPn)
{
    int n = nx * ny * nz;
    
    dim3 nt1(16,16);
    int nb1 = (n-1) / 16 + 1;
    int nt2 = nx - 4;
    int nb2 = ny - 4;
    
    
    // First time initialization.
    static bool firstTime = true;
    if(firstTime) {
        firstTime = false;
        cudaInit(nx, ny, nz, nm, nq, w, p, xi, eta, mu);
    }
    
    
    // Do the computations.
    cudaMemcpy(alphaDev, alpha, n * nm * sizeof(double), cudaMemcpyHostToDevice);  CUDA_ERROR_SCRIPT
    cudaMemset(fluxDev, 0, n * nm * sizeof(double));  CUDA_ERROR_SCRIPT
    for(int q = 0; q < nq; q++) {
        kernel_computeAnsatz<<<nb1, nt1>>>(n, nm, q, alphaDev, pDev, ansatzDev, isPn);
//         double alpha = 1.0;
//         double beta = 0.0;
//         cublasDgemv(handle, CUBLAS_OP_T, nm, n, 
//                     &alpha, alphaDev, nm, &pDev[q*nm], 1, &beta, ansatzDev, 1);
        
        kernel_solveFluxTemp<<<nb2, nt2>>>(nx, ny, nz, q, dx, dy, dz, 
                                           xiDev, etaDev, muDev, wDev, 
                                           ansatzDev, fluxTempDev);
        kernel_solveFlux<<<nb1, nt1>>>(nm, q, fluxTempDev, pDev, fluxDev);
    }
    cudaMemcpy(flux, fluxDev, n * nm * sizeof(double), cudaMemcpyDeviceToHost);  CUDA_ERROR_SCRIPT
}
