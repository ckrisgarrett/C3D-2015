/*
    File:   opt.cpp
    Author: Kris Garrett
    Date:   February 13, 2013
*/

#include "opt.h"
#include "../utils.h"
#include "../global.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>


/*
    Dumps out debugging data to file.
*/
static void dumpData(int nm, int nm2, int nq, double t, double r, int iter, 
                     int iterGamma, int regIndex, double *u, double *alpha, 
                     double *uOriginal, double *alpha0, double *h)
{
    FILE *file;
    char filename[100];
    
    sprintf(filename, "DataDump_%d.dat", g_node);
    file = fopen(filename, "w");
    if(file == NULL) {
        printf("Could not open file to output data.\n");
        return;
    }
    
    
    fprintf(file, "--- GLOBAL DATA ---\n");
    fprintf(file, "g_node = %d\n", g_node);
    fprintf(file, "g_numNodes = %d\n", g_numNodes);
    fprintf(file, "g_solver = %d\n", g_solver);
    fprintf(file, "g_initCond = %d\n", g_initCond);
    fprintf(file, "g_nx = %d  g_ny = %d  g_nz = %d\n", g_nx, g_ny, g_nz);
    fprintf(file, "g_globalNx = %d  g_globalNy = %d  g_globalNz = %d\n", g_globalNx, g_globalNy, g_globalNz);
    fprintf(file, "g_mpix = %d  g_mpiy = %d  g_mpiz = %d\n", g_mpix, g_mpiy, g_mpiz);
    fprintf(file, "g_ax = %e  g_ay = %e  g_az = %e\n", g_ax, g_ay, g_az);
    fprintf(file, "g_bx = %e  g_by = %e  g_bz = %e\n", g_bx, g_by, g_bz);
    fprintf(file, "g_globalAx = %e  g_globalAy = %e  g_globalAz = %e\n", g_globalAx, g_globalAy, g_globalAz);
    fprintf(file, "g_globalBx = %e  g_globalBy = %e  g_globalBz = %e\n", g_globalBx, g_globalBy, g_globalBz);
    fprintf(file, "g_numOmpThreads = %d\n", g_numOmpThreads);
    fprintf(file, "g_dx = %e  g_dy = %e  g_dz = %e\n", g_dx, g_dy, g_dz);
    fprintf(file, "g_nodeX = %d  g_nodeY = %d  g_nodeZ = %d\n", g_nodeX, g_nodeY, g_nodeZ);
    fprintf(file, "g_numQuadPoints = %d\n", g_numQuadPoints);
    fprintf(file, "g_numMoments = %d\n", g_numMoments);
    fprintf(file, "g_numMoments2 = %d\n", g_numMoments2);
    fprintf(file, "g_pnUseKineticFlux = %d\n", g_pnUseKineticFlux);
    fprintf(file, "g_gX[0] = %d  g_gX[1] = %d  g_gX[2] = %d  g_gX[3] = %d\n", g_gX[0], g_gX[1], g_gX[2], g_gX[3]);
    fprintf(file, "g_gY[0] = %d  g_gY[1] = %d  g_gY[2] = %d  g_gY[3] = %d\n", g_gY[0], g_gY[1], g_gY[2], g_gY[3]);
    fprintf(file, "g_gZ[0] = %d  g_gZ[1] = %d  g_gZ[2] = %d  g_gZ[3] = %d\n", g_gZ[0], g_gZ[1], g_gZ[2], g_gZ[3]);
    fprintf(file, "g_tFinal = %e\n", g_tFinal);
    fprintf(file, "g_outDt = %e\n", g_outDt);
    fprintf(file, "g_maxDt = %e\n", g_maxDt);
    fprintf(file, "g_theta = %e\n", g_theta);
    fprintf(file, "g_mnMaxIter = %d\n", g_mnMaxIter);
    fprintf(file, "g_mnTol = %e\n", g_mnTol);
    fprintf(file, "g_mnCondHMax = %e\n", g_mnCondHMax);
    fprintf(file, "g_mnUseGaunt = %d\n", g_mnUseGaunt);
    fprintf(file, "g_mnBatched = %d\n", g_mnBatched);
    fprintf(file, "g_mnTolGamma = %e\n", g_mnTolGamma);
    fprintf(file, "g_cudaKinetic = %d\n", g_cudaKinetic);
    fprintf(file, "g_cudaBatched = %d\n", g_cudaBatched);
    fprintf(file, "g_cudaBatchSize = %d\n", g_cudaBatchSize);
    
    fprintf(file, "--- OPTIMIZATION DATA ---\n");
    fprintf(file, "c_t = %e\n", t);
    fprintf(file, "c_r = %e\n", r);
    fprintf(file, "c_iter = %d\n", iter);
    fprintf(file, "c_iterGamma = %d\n", iterGamma);
    fprintf(file, "c_regIndex = %d\n", regIndex);
    
    fprintf(file, "u = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", u[i]);
    fprintf(file, "alpha = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", alpha[i]);
    fprintf(file, "uOriginal = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", uOriginal[i]);
    fprintf(file, "alpha0 = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", alpha0[i]);
    fprintf(file, "h = \n");
    for(int i = 0; i < nm; i++) {
        fprintf(file, "    ");
        for(int j = 0; j < nm; j++) {
            fprintf(file, "%e ", h[i*nm+j]);
        }
        fprintf(file, "\n");
    }
    
    
    fclose(file);
}


/*
    Calculate alpha for u = (1,0,0,...,0)^T
*/
static 
void isotropicAlpha(int n, double *alpha)
{
    alpha[0] = 2.0 * sqrt(M_PI) * log(0.5 / sqrt(M_PI));
    for(int i = 1; i < n; i++)
        alpha[i] = 0.0;
}


/*
    Estimate an upper bound for the error.
*/
static 
double modcfl(int n, double *da)
{
    double da_1;
    int pnOrder;
    double m_inf;
    
    da_1 = 0.0;
    for(int i = 0; i < n; i++)
        da_1 = da_1 + fabs(da[i]);
    
    pnOrder = floor(sqrt(n)) + 1;
    m_inf = sqrt((2 * pnOrder + 1) / (2 * M_PI));
    
    return exp(2.0 * da_1 * m_inf);
}



/*
    Does the optimization for u = (1,...)^T
*/
static 
void optScaled(int nm, int nm2, int nq, double *u, double *alpha, double *w, double *p, double *p2, 
               OPTIONS *options, OUTS *outs)
{
    // Constants
    int inc1 = 1;
    char lChar = 'L';
    int maxIter = options->maxIter;
    double tolRel = options->tolRel;
    double tolGamma = options->tolGamma;
    double condHMax = options->condHMax;
    
    // Variables
    double f;           // variable to minimize
    double r;           // regularization amount
    int regIndex;       // regularization index
    double rtol;        // relative tolerance for calculations
    int iter;           // number of iterations since last regularization
    int totalIter;      // total number of iterations including regularization
    int iterGamma;      // number of iterations past tolerance success but gamma tolerance failure
    double t;           // part of alpha = alpha + t*d
    double gamma;       // another criteria for convergence (converges to 1)
    double err;         // error in gradient
    int choleskyError;  // error for computing cholesky
    int conditionError; // error for computing condition number
    double norm1;       // 1-norm of h
    double condest;     // condition number estimate for h
    
    // Static variables to allocate only once
    static double *g;           // gradient
    static double *h;           // hessian
    static double *d;           // search direction
    static double *cholesky;    // cholesky factorization of h
    static double *uOriginal;   // input u
    static double *alpha0;      // input alpha
    static double *work;        // scratch work needed to compute condition number
    static int *iwork;          // scratch work needed to compute condition number
    static bool firstTime = true;
    #pragma omp threadprivate(g,h,d,cholesky,uOriginal,alpha0,work,iwork,firstTime)
    
    
    // First time allocation of variables.
    if(firstTime) {
        firstTime = false;
        g = (double*)malloc(nm * sizeof(double));
        h = (double*)malloc(nm * nm * sizeof(double));
        d = (double*)malloc(nm * sizeof(double));
        cholesky = (double*)malloc(nm * nm * sizeof(double));
        uOriginal = (double*)malloc(nm * sizeof(double));
        alpha0 = (double*)malloc(nm * sizeof(double));
        work = (double*)malloc(3*nm*sizeof(double));
        iwork = (int*)malloc(nm*sizeof(int));
    }
    
    
    // Initialize variables
    rtol = dnrm2_(&nm, u, &inc1) * tolRel;
    iter = 0;
    totalIter = 0;
    iterGamma = 0;
    t = 1;
    r = 0.0;
    regIndex = 0;
    
    
    // Save original data
    for(int i = 0; i < nm; i++)
        alpha0[i] = alpha[i];
    
    
    for(int i = 0; i < nm; i++)
        uOriginal[i] = u[i];
    
    
    // Iterate to find alpha.
    while(true) {
        // increment the iteration counters
        iter = iter + 1;
        totalIter = totalIter + 1;
        
        
        // Find f, g, and h.
        f = fobjMn(nm, nm2, nq, alpha, u, w, p, p2, g, h, options->useGaunt, options->useGauntSparse);
        
        
        // Check quadrature and find Newton direction with Cholesky factorization.
        // ALSO NEED TO CHECK CONDITION NUMBER OF H.
        memcpy(cholesky, h, nm * nm * sizeof(double));
        dpotrf_(&lChar, &nm, cholesky, &nm, &choleskyError);
        
        norm1 = utils_norm1(nm, h);
        dpocon_(&lChar, &nm, cholesky, &nm, &norm1, &condest, work, iwork, &conditionError);
        
        condest = 1.0 / condest;
        
        
        // Initialize to isotropic alpha if H is too bad on first iteration.
        if(totalIter == 1 && (choleskyError != 0 || condest > condHMax)) {
            isotropicAlpha(nm, alpha0);
            
            // Reset Variables.
            rtol = dnrm2_(&nm, u, &inc1) * tolRel;
            memcpy(alpha, alpha0, nm * sizeof(double));
            iter = 0;
            iterGamma = 0;
            t = 1;
            
            continue;
        }
        
        
        // Regularize criteria.
        if(iter >= maxIter || choleskyError != 0 || t == 0 || condest > condHMax) {
            // REGULARIZE ...
            if(regIndex < NUM_REGULARIZATIONS - 1) {
                regIndex++;
                r = REGULARIZATIONS[regIndex];
                
                u[0] = 1.0;
                for(int i = 1; i < nm; i++)
                    u[i] = uOriginal[i] * (1.0 - r);
            }
            else {
                // This should never happen.
                printf("Max regularization achieved.\n");
                dumpData(nm, nm2, nq, t, r, iter, iterGamma, regIndex, u, alpha, uOriginal, alpha0, h);
                utils_abort();
            }
            
            
            // Reset variables.
            rtol = dnrm2_(&nm, u, &inc1) * tolRel;
            memcpy(alpha, alpha0, nm * sizeof(double));
            iterGamma = 0;
            iter = 0;
            t = 1;
            if(regIndex == NUM_REGULARIZATIONS - 1)
                maxIter = 10 * maxIter;
            
            continue;
        }
        
        
        // Solve for line direction.
        for(int i = 0; i < nm; i++)
            d[i] = g[i];
        dpotrs_(&lChar, &nm, &inc1, cholesky, &nm, d, &nm, &choleskyError);
        for(int i = 0; i < nm; i++)
            d[i] = -d[i];
        
        
        // the error as measured by the norm of the gradient
        err = dnrm2_(&nm, g, &inc1);
        
        // gamma, the supplementary stopping criterion
        gamma = modcfl(nm, d);
        
        
        // now check the stopping criteria
        if(err <= rtol && gamma <= tolGamma) {
            // problem solved!
            break;
        }
        else if(err <= rtol && gamma > tolGamma) {
            // this is to keep track of how many extra iterations we took just for gamma
            iterGamma++;
        }
        
        
        // if you get to this point, that means you didn't satisfy the stopping
        // criteria and you didn't need to regularize more than you already are
        // so, it's time to take a step
        t = linesearch(nm, nm2, nq, alpha, d, f, g, u, w, p);
    }
    
    
    if(outs != NULL) {
        outs->iter = totalIter;
        outs->iterGamma = iterGamma;
        if(outs->iterGamma < 0)
            outs->iterGamma = 0;
        outs->gamma = gamma;
        outs->normG = err;
        outs->r = r;
    }
}


/*
    Scales u and calls the optimization routine.
*/
void opt(int nm, int nm2, int nq, double *u, double *alpha, OPTIONS options, double *w, double *p, 
         double *p2, OUTS *outs)
{
    double u0, alphaScale;
    
    // Scale u and alpha.
    u0 = u[0];
    alphaScale = log(u0) * 2.0 * sqrt(M_PI);
    alpha[0] = alpha[0] - alphaScale;
    
    for(int i = 0; i < nm; i++)
        u[i] = u[i] / u0;
    
    if(u0 <= 0) {
        printf("u[0] < 0\n");
        utils_abort();
    }
    
    // Run optimization.
    optScaled(nm, nm2, nq, u, alpha, w, p, p2, &options, outs);

    // Rescale u and alpha.
    for(int i = 0; i < nm; i++)
        u[i] = u[i] * u0;
    alpha[0] = alpha[0] + alphaScale;
}

