/*
    File:   linesearch.cpp
    Author: Kris Garrett
    Date:   February 13, 2013
*/

#include "opt.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>


static double *alphaOld;
#pragma omp threadprivate(alphaOld)


void linesearch_init(int nm)
{
    #pragma omp parallel
    {
        alphaOld = (double*)malloc(nm * sizeof(double));
    }
}


/*
    Find alpha which decreases f.
*/
double linesearch(int n1, int n2, int nq, double *alpha, double *d, double f, double *g, 
                  double *u, double *w, double *p)
{
    double t = 1.0;
    double fNew = f;
    double delta = 1e-3; 
    double beta = .5;       // decrease rate of step size
    double maxFactor = 0;
    double temp;
    
    
    // Copy alpha.
    for(int i = 0; i < n1; i++)
        alphaOld[i] = alpha[i];
    
    
    // Create factor to check for backtracking.
    for(int i = 0; i < n1; i++) {
        temp = fabs(d[i]) / (fabs(alphaOld[i]) + 1e-100);
        if(temp > maxFactor)
            maxFactor = temp;
    }
    
    
    // backtracking
    while(t * maxFactor > DBL_EPSILON / 2) {
        // alpha = alphaOld + t*d;
        for(int i = 0; i < n1; i++)
            alpha[i] = alphaOld[i] + t * d[i];
        
        fNew = fobjMn(n1, n2, nq, alpha, u, w, p, NULL, NULL, NULL, false, false);
        
        // check the Armijo criterion with the current quadrature
        temp = 0.0;
        for(int i = 0; i < n1; i++)
            temp += g[i] * d[i];
        if(fNew <= f + delta * t * temp)
            return t;
        
        t = beta * t;
    }
    
    
    // Getting to here means we backtracked all the way.
    // if we backtracked all the way back to the starting point, here we make
    // sure to send back evaluations of the objective function and gradient
    // using the (possibly) new quadrature
    for(int i = 0; i < n1; i++)
        alpha[i] = alphaOld[i];
    return 0.0;
}


