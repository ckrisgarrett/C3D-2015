/*
    File:   fobj.cpp
    Author: Kris Garrett
    Date:   May 25, 2012
*/

#include "opt.h"
#include <math.h>
#include <stdlib.h>


double *g_beta;
double *g_betaVal;
int *g_betaColInd;
int *g_betaRowPtr;
static double *gExtended;
#pragma omp threadprivate(gExtended)


void fobj_init(int nm2) 
{
    #pragma omp parallel
    {
        gExtended = (double*)malloc(nm2 * sizeof(double));
    }
}



/*
    Solves for the Hessian for the Mn algorithm.
*/
//__attribute__ ((noinline))  // This is here to aid in profiling.
void solveHMn(int nm, int nm2, int nq, double *alpha, double *w, double *p, double *p2, 
              double *h, bool useGaunt, bool useGauntSparse)
{
    double temp, pi, pj, h_ij;
    
    
    // Zero variables.
    for(int k = 0; k < nm2; k++)
        gExtended[k] = 0.0;
    
    for(int i = 0; i < nm*nm; i++)
        h[i] = 0.0;
    
    
    // Integrate.
    for(int q = 0; q < nq; q++) {
        // Compute temp = exp(alpha^T * p).
        temp = 0;
        for(int i = 0; i < nm; i++)
            temp = temp + alpha[i] * p[q*nm+i];
        temp = exp(temp);

        
        // Compute H.
        if(useGaunt) {
            for(int k = 0; k < nm2; k++)
                gExtended[k] += w[q] * p2[q*nm2+k] * temp;
        }
        else {
            for(int i = 0; i < nm; i++) {
                pi = p[q*nm+i];
                for(int j = 0; j <= i; j++) {
                    pj = p[q*nm+j];
                    h[i*nm+j] += w[q] * pi * pj * temp;
                }
            }
        }
    }
    
    
    // Finish the computations.
    if(useGaunt && !useGauntSparse) {
        for(int i = 0; i < nm; i++) {
            for(int j = i; j < nm; j++) {
                h_ij = 0.0;
                for(int k = 0; k < nm2; k++)
                    h_ij += gExtended[k] * g_beta[(i*nm+j)*nm2+k];
                h[i*nm+j] = h[j*nm+i] = h_ij;
            }
        }
    }
    else if(useGaunt && useGauntSparse) {
        for(int i = 0; i < nm; i++) {
            for(int j = i; j < nm; j++) {
                h_ij = 0.0;
                int index = i * nm + j;
                for(int k = g_betaRowPtr[index]; k < g_betaRowPtr[index+1]; k++)
                    h_ij += gExtended[g_betaColInd[k]] * g_betaVal[k];
                h[i*nm+j] = h[j*nm+i] = h_ij;
            }
        }
    }
    else {
        for(int i = 0; i < nm; i++) {
            for(int j = i+1; j < nm; j++)
                h[i*nm+j] = h[j*nm+i];
        }
    }
}


/*
    Evaluates f = integral(e^(alpha^T p)) - alpha^T u
              g = integral(p e^(alpha^T p)) - u
              h = integral(p p^T e^(alpha^T p))
*/
double fobjMn(int nm, int nm2, int nq, double *alpha, double *u, double *w, double *p, double *p2, 
              double *g, double *h, bool useGaunt, bool useGauntSparse)
{
    double f;
    double temp;
    
    
    // If H != NULL, then get all the information for f and g from H.
    if(h != NULL) {
        solveHMn(nm, nm2, nq, alpha, w, p, p2, h, useGaunt, useGauntSparse);
        
        if(g != NULL) {
            for(int i = 0; i < nm; i++)
                g[i] = h[i] / p[0] - u[i];
        }
        
        f = h[0] / (p[0] * p[0]);
        for(int i = 0; i < nm; i++)
            f = f - alpha[i] * u[i];
        
        return f;
    }
    
    
    // If H == NULL, then find g and f.
    if(g != NULL) {
        for(int i = 0; i < nm; i++)
            g[i] = 0.0;
    }
    
    f = 0.0;
    for(int q = 0; q < nq; q++) {
        // Compute temp = exp(alpha^T * p).
        temp = 0;
        for(int i = 0; i < nm; i++)
            temp = temp + alpha[i] * p[q*nm+i];
        temp = exp(temp);
        
        if(g != NULL) {
            for(int i = 0; i < nm; i++)
                g[i] += w[q] * p[q*nm+i] * temp;
        }

        f = f + w[q] * temp;
    }
    
    
    if(g != NULL) {
        for(int i = 0; i < nm; i++)
            g[i] = g[i] - u[i];
    }
    for(int i = 0; i < nm; i++) 
        f = f - alpha[i] * u[i];
    
    return f;
}


/*
    Calculates g and f from H.
*/
double fobjMnUseH(int nm, double *alpha, double *u, double *p, double *g, double *h)
{
    double f = 0;
    
    if(g != NULL) {
        for(int i = 0; i < nm; i++)
            g[i] = h[i] / p[0] - u[i];
    }
    
    f = h[0] / (p[0] * p[0]);
    for(int i = 0; i < nm; i++)
        f = f - alpha[i] * u[i];
    
    return f;
}
