/*
    File:   optBatch.cpp
    Author: Kris Garrett
    Date:   September 17, 2013
*/

#include "opt.h"
#include "../utils.h"
#include "../cuda/cuda_headers.h"
#include "../global.h"
#include "../timer.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>


// Optimization class static variables.
std::stack<Optimization*> *Optimization::c_stack1 = NULL;
std::stack<Optimization*> *Optimization::c_stack2 = NULL;
std::stack<Optimization*> *Optimization::c_stack3 = NULL;


/*
    Dumps out debugging data to file.
*/
static void dumpData(int nm, int nm2, int nq, Optimization *opt)
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
    fprintf(file, "c_t = %e\n", opt->c_t);
    fprintf(file, "c_r = %e\n", opt->c_r);
    fprintf(file, "c_u0 = %e\n", opt->c_u0);
    fprintf(file, "c_iter = %d\n", opt->c_iter);
    fprintf(file, "c_iterGamma = %d\n", opt->c_iterGamma);
    fprintf(file, "c_totalIter = %d\n", opt->c_totalIter);
    fprintf(file, "c_i = %d  c_j = %d  c_k = %d\n", opt->c_i, opt->c_j, opt->c_k);
    fprintf(file, "c_regIndex = %d\n", opt->c_regIndex);
    fprintf(file, "c_preProcessed = %d\n", opt->c_preProcessed);
    
    fprintf(file, "c_u = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", opt->c_u[i]);
    fprintf(file, "c_alpha = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", opt->c_alpha[i]);
    fprintf(file, "c_uOriginal = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", opt->c_uOriginal[i]);
    fprintf(file, "c_alpha0 = \n");
    for(int i = 0; i < nm; i++)
        fprintf(file, "    %e\n", opt->c_alpha0[i]);
    fprintf(file, "c_h = \n");
    for(int i = 0; i < nm; i++) {
        fprintf(file, "    ");
        for(int j = 0; j < nm; j++) {
            fprintf(file, "%e ", opt->c_h[i*nm+j]);
        }
        fprintf(file, "\n");
    }
    
    
    fclose(file);
}


/*
    Calculate alpha for u = (1,0,0,...,0)^T
*/
static void isotropicAlpha(int n, double *alpha)
{
    alpha[0] = 2.0 * sqrt(M_PI) * log(0.5 / sqrt(M_PI));
    for(int i = 1; i < n; i++)
        alpha[i] = 0.0;
}


/*
    Estimate an upper bound for the error.
*/
static double modcfl(int n, double *da)
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


static
bool solveTheRest(int nm, int nm2, int nq, Optimization *opt)
{
    // Constants
    int inc1 = 1;
    char lChar = 'l';
    int maxIter = g_mnMaxIter;
    
    // Variables
    double f;           // variable to minimize
    double gamma;       // another criteria for convergence (converges to 1)
    int choleskyError;  // error for computing cholesky
    int conditionError; // error for computing condition number
    double norm1;       // 1-norm of h
    double condest;     // condition number estimate for h
    
    
    
    // Static variables to allocate only once
    static double *g;           // gradient
    static double *d;           // search direction
    static double *cholesky;    // cholesky factorization of h
    static double *work;        // scratch work needed to compute condition number
    static int *iwork;          // scratch work needed to compute condition number
    static bool firstTime = true;
    #pragma omp threadprivate(d,cholesky,g,work,iwork,firstTime)
    
    
    // First time allocation of variables.
    if(firstTime) {
        firstTime = false;
        d = (double*)malloc(nm * sizeof(double));
        cholesky = (double*)malloc(nm * nm * sizeof(double));
        g = (double*)malloc(nm * sizeof(double));
        work = (double*)malloc(3*nm*sizeof(double));
        iwork = (int*)malloc(nm*sizeof(int));
    }
    
    
    // Initialize variables
    if(opt->c_regIndex == NUM_REGULARIZATIONS - 1)
        maxIter = 10 * maxIter;
    
    
    // increment the iteration counters
    opt->c_iter = opt->c_iter + 1;
    opt->c_totalIter = opt->c_totalIter + 1;
    
    
    // Find f, g, and h.
    f = fobjMnUseH(nm, opt->c_alpha, opt->c_u, g_m, g, opt->c_h);
    
    
    // Check quadrature and find Newton direction with Cholesky factorization.
    // ALSO NEED TO CHECK CONDITION NUMBER OF H.
    memcpy(cholesky, opt->c_h, nm * nm * sizeof(double));
    dpotrf_(&lChar, &nm, cholesky, &nm, &choleskyError);
    
    norm1 = utils_norm1(nm, opt->c_h);
    dpocon_(&lChar, &nm, cholesky, &nm, &norm1, &condest, work, iwork, &conditionError);
    
    condest = 1.0 / condest;


    // Initialize to isotropic alpha if H is too bad on first iteration.
    if(opt->c_totalIter == 1 && (choleskyError != 0 || condest > g_mnCondHMax)) {
        isotropicAlpha(nm, opt->c_alpha0);
        
        // Reset Variables.
        memcpy(opt->c_alpha, opt->c_alpha0, nm * sizeof(double));
        opt->c_iter = 0;
        opt->c_iterGamma = 0;
        opt->c_t = 1;
        
        return false;
    }
    
    
    // Regularize criteria.
    if(opt->c_iter >= maxIter || choleskyError != 0 || opt->c_t == 0 || 
       condest > g_mnCondHMax) {
        // REGULARIZE ...
        if(opt->c_regIndex < NUM_REGULARIZATIONS - 1) {
            opt->c_regIndex++;
            opt->c_r = REGULARIZATIONS[opt->c_regIndex];
            
            opt->c_u[0] = 1.0;
            for(int k = 1; k < nm; k++)
                opt->c_u[k] = opt->c_uOriginal[k] * (1.0 - opt->c_r);
        }
        else {
            // This should never happen.
            printf("Max regularization achieved.\n");
            dumpData(nm, nm2, nq, opt);
            utils_abort();
        }
        
        
        // Reset variables.
        memcpy(opt->c_alpha, opt->c_alpha0, nm * sizeof(double));
        opt->c_iterGamma = 0;
        opt->c_iter = 0;
        opt->c_t = 1;
        
        return false;
    }
    
    
    // Solve for line direction.
    for(int i = 0; i < nm; i++)
        d[i] = g[i];
    //dpotrs_(&lChar, &nm, &inc1, cholesky, &nm, d, &nm, &choleskyError);
    dpotrs(nm, cholesky, d);
    for(int i = 0; i < nm; i++)
        d[i] = -d[i];
    
    
    // the error as measured by the norm of the gradient
    opt->c_err = dnrm2_(&nm, g, &inc1);
    
    // gamma, the supplementary stopping criterion
    gamma = modcfl(nm, d);
    
    
    // now check the stopping criteria
    if(opt->c_err <= g_mnTol && gamma <= g_mnTolGamma) {
        // problem solved!
        return true;
    }
    else if(opt->c_err <= g_mnTol && gamma > g_mnTolGamma) {
        // this is to keep track of how many extra iterations we took just for gamma
        opt->c_iterGamma++;
    }
    
    
    // if you get to this point, that means you didn't satisfy the stopping
    // criteria and you didn't need to regularize more than you already are
    // so, it's time to take a step
    opt->c_t = linesearch(nm, nm2, nq, opt->c_alpha, d, f, g, opt->c_u, g_w, g_m);
    return false;
}





Optimization::Optimization(int i, int j, int k, int n, double *u, double *alpha)
{
    c_i = i;
    c_j = j;
    c_k = k;
    c_u = (double*)malloc(n * sizeof(double));
    memcpy(c_u, u, n * sizeof(double));
    c_alpha = (double*)malloc(n * sizeof(double));
    memcpy(c_alpha, alpha, n * sizeof(double));
    c_h = NULL;
    c_alpha0 = NULL;
    c_uOriginal = NULL;
    c_t = 1;
    c_r = 0.0;
    c_iter = 0;
    c_iterGamma = 0;
    c_totalIter = 0;
    c_regIndex = 0;
    c_u0 = c_u[0];
    c_preProcessed = false;
}


void Optimization::preProcess(int n)
{
    double alphaScale = log(c_u0) * 2.0 * sqrt(M_PI);
    c_alpha[0] = c_alpha[0] - alphaScale;
    
    for(int k = 0; k < n; k++)
        c_u[k] = c_u[k] / c_u0;
    if(c_u0 <= 0)
    {
        printf("u[0] < 0\n");
        utils_abort();
    }
    
    c_h = (double*)malloc(n * n * sizeof(double));
    c_alpha0 = (double*)malloc(n * sizeof(double));
    memcpy(c_alpha0, c_alpha, n * sizeof(double));
    c_uOriginal = (double*)malloc(n * sizeof(double));
    memcpy(c_uOriginal, c_u, n * sizeof(double));
    c_preProcessed = true;
}


void Optimization::postProcess(int n)
{
    double alphaScale = log(c_u0) * 2.0 * sqrt(M_PI);
    for(int k = 0; k < n; k++)
        c_u[k] = c_u[k] * c_u0;
    c_alpha[0] = c_alpha[0] + alphaScale;
    
    free(c_alpha0);
    free(c_h);
    free(c_uOriginal);
}


/*
    
*/
void Optimization::opt(int nm, int nm2, int nq)
{
    #ifdef USE_CUDA
    static bool firstTime = true;
    static double *alphas;
    static double *hs;
    static Optimization **intermediateQueue;
    if(firstTime)
    {
        firstTime = false;
        
        alphas = (double*)malloc(g_cudaBatchSize * nm * sizeof(double));
        hs = (double*)malloc(g_cudaBatchSize * nm * nm * sizeof(double));
        intermediateQueue = (Optimization**)malloc(g_cudaBatchSize * sizeof(Optimization*));
        solveHInit_cuda(nm, nm2, nq, g_cudaBatchSize, g_w, g_m, g_m2, g_beta, 
                        g_betaVal, g_betaColInd, g_betaRowPtr, g_mnUseGaunt);
    }
    #endif
    
    
    Timer t1, t2;
    while(!(c_stack1->empty() && c_stack2->empty()))
    {
        t1.start();
        // Use CUDA to solve for H.
        #ifdef USE_CUDA
        if(g_cudaBatched) {
            int numBatched = 0;
            for(int i = 0; i < g_cudaBatchSize; i++)
            {
                if(!c_stack1->empty())
                {
                    Optimization *opt = c_stack1->top();
                    c_stack1->pop();
                    if(!opt->c_preProcessed)
                        opt->preProcess(nm);
                    
                    memcpy(&alphas[i * nm], opt->c_alpha, nm * sizeof(double));
                    intermediateQueue[i] = opt;
                    numBatched++;
                }
            }

            solveH_cuda(nm, nm2, nq, g_cudaBatchSize, alphas, hs, g_mnUseGaunt, g_mnUseGauntSparse);
            
            for(int i = 0; i < numBatched; i++)
            {
                Optimization *opt = intermediateQueue[i];
                memcpy(opt->c_h, &hs[i * nm * nm], nm * nm * sizeof(double));
                c_stack2->push(opt);
            }
        }
        else {
            // Use CPU to solve for H.
            #endif
            #pragma omp parallel for
            for(int i = 0; i < g_cudaBatchSize; i++)
            {
                Optimization *opt = NULL;
                #pragma omp critical
                {
                    if(!c_stack1->empty())
                    {
                        opt = c_stack1->top();
                        c_stack1->pop();
                    }
                }
                if(opt != NULL)
                {
                    if(!opt->c_preProcessed)
                        opt->preProcess(nm);
                    solveHMn(nm, nm2, nq, opt->c_alpha, g_w, g_m, g_m2, opt->c_h, 
                             g_mnUseGaunt, g_mnUseGauntSparse);
                    #pragma omp critical
                    {
                        c_stack2->push(opt);
                    }
                }
            }
        #ifdef USE_CUDA
        }
        #endif
        t1.stop();
        t2.start();
        // Do rest of optimization.
        #pragma omp parallel for
        for(int i = 0; i < g_cudaBatchSize; i++)
        {
            Optimization *opt = NULL;
            #pragma omp critical
            {
                if(!c_stack2->empty())
                {
                    opt = c_stack2->top();
                    c_stack2->pop();
                }
            }
            if(opt != NULL)
            {
                bool finished = solveTheRest(nm, nm2, nq, opt);
                if(finished)
                {
                    opt->postProcess(nm);
                    #pragma omp critical
                    {
                        c_stack3->push(opt);
                    }
                }
                else
                {
                    #pragma omp critical
                    {
                        c_stack1->push(opt);
                    }
                }
            }
        }
        t2.stop();
    }
    if(g_node == 0) 
        printf("   (BATCH) Hessian timer 1: %.3lfs   OtherOpt timer 2: %.3lfs\n", 
                t1.getTimeElapsed(), t2.getTimeElapsed());
}

