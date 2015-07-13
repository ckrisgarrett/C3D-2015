/*
    File:   update.cpp
    Author: Kris Garrett
    Date:   September 4, 2013
*/

#include "utils.h"
#include "timer.h"
#include "global.h"
#include "opt/opt.h"
#include <omp.h>
#include <strings.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


/*
    Does an Euler step.
*/
static
void eulerStep(double *u, double *flux, double dt)
{
    #pragma omp parallel for
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        u[IU(i,j,k,0)] = u[IU(i,j,k,0)] * 
            (1.0 + dt * g_sigmaS[I3D(i,j,k)] - dt * g_sigmaT[I3D(i,j,k)]) - 
            dt * flux[IU(i,j,k,0)];
        
        for(int m = 1; m < g_numMoments; m++) {
            u[IU(i,j,k,m)] = u[IU(i,j,k,m)] * 
                (1.0 - dt * g_sigmaT[I3D(i,j,k)]) - dt * flux[IU(i,j,k,m)];
        }
    }}}
}


/*
    Solves the optimization problem on the entire grid.
*/
static
void solveOptimization(int integratorIndex)
{
    OPTIONS options;
    OUTS outs;
    int optIndex;
    
    #pragma omp parallel for private(options,outs,optIndex)
    for(int i = g_gX[0]; i <= g_gX[3]; i++) {
    for(int j = g_gY[0]; j <= g_gY[3]; j++) {
    for(int k = g_gZ[0]; k <= g_gZ[3]; k++) {
        options.maxIter = g_mnMaxIter;
        options.tolAbs = g_mnTol * g_mnTol;
        options.tolRel = g_mnTol;
        options.tolGamma = 1.1;
        options.condHMax = g_mnCondHMax;
        options.useGaunt = g_mnUseGaunt;
        options.useGauntSparse = g_mnUseGauntSparse;
        
        opt(g_numMoments, g_numMoments2, g_numQuadPoints, &g_u[IU(i,j,k,0)], 
            &g_mnAlpha[IU(i,j,k,0)], options, g_w, g_m, g_m2, &outs);
        
        
        if(g_outputOptStats && 
           j == g_gY[3] / 2 && k == g_gZ[3] / 2 && i >= g_gX[1] && i <= g_gX[2]) {
            
            optIndex = (i - g_gX[1]) + g_nx * (2 * g_numUpdates + integratorIndex);
            
            g_optStats[optIndex].iter = outs.iter;
            g_optStats[optIndex].iterGamma = outs.iterGamma;
            g_optStats[optIndex].r = outs.r;
            g_optStats[optIndex].gamma = outs.gamma;
            g_optStats[optIndex].normG = outs.normG;
        }
        
        
        #pragma omp critical
        {
            if(outs.iter > g_mnMaxIterHist)
                g_mnMaxIterHist = outs.iter;
            if(outs.iterGamma > g_mnMaxGammaIter)
                g_mnMaxGammaIter = outs.iterGamma;
            for(int r = 0; r < NUM_REGULARIZATIONS; r++) {
                if(fabs(outs.r - REGULARIZATIONS[r]) < 1e-10)
                    g_mnHistReg[r]++;
            }
            
            g_mnIterMean = 
                (g_mnNumDualSolves * g_mnIterMean + outs.iter) / 
                (g_mnNumDualSolves + 1.0);
            g_mnIterGammaMean = 
                (g_mnNumDualSolves * g_mnIterGammaMean + outs.iterGamma) / 
                (g_mnNumDualSolves + 1.0);
            g_mnNumDualSolves++;
        }
    }}}
}


/*
    Solves the optimization problem on the entire grid in batches.
*/
static 
void solveBatchOptimization(int integratorIndex)
{
    Optimization *opt;
    int i,j,k;
    static bool firstTime = true;
    int optIndex;
    
    if(firstTime) {
        firstTime = false;
        Optimization::c_stack1 = new std::stack<Optimization*>;
        Optimization::c_stack2 = new std::stack<Optimization*>;
        Optimization::c_stack3 = new std::stack<Optimization*>;
    }
    
    
    for(i = g_gX[0]; i <= g_gX[3]; i++) {
    for(j = g_gY[0]; j <= g_gY[3]; j++) {
    for(k = g_gZ[0]; k <= g_gZ[3]; k++) {
        opt = new Optimization(i, j, k, g_numMoments, &g_u[IU(i,j,k,0)], 
                               &g_mnAlpha[IU(i,j,k,0)]);
        Optimization::c_stack1->push(opt);
    }}}
    
    
    Optimization::opt(g_numMoments, g_numMoments2, g_numQuadPoints);
    
    
    while(!Optimization::c_stack3->empty()) {
        opt = Optimization::c_stack3->top();
        Optimization::c_stack3->pop();
        i = opt->c_i;
        j = opt->c_j;
        k = opt->c_k;
        memcpy(&g_mnAlpha[IU(i,j,k,0)], opt->c_alpha, g_numMoments * sizeof(double));
        free(opt->c_alpha);
        memcpy(&g_u[IU(i,j,k,0)], opt->c_u, g_numMoments * sizeof(double));
        free(opt->c_u);
        
        
        if(g_outputOptStats && 
           j == g_gY[3] / 2 && k == g_gZ[3] / 2 && i >= g_gX[1] && i <= g_gX[2]) {
            
            optIndex = (i - g_gX[1]) + g_nx * (2 * g_numUpdates + integratorIndex);
            
            g_optStats[optIndex].iter = opt->c_totalIter;
            g_optStats[optIndex].iterGamma = opt->c_iterGamma;
            g_optStats[optIndex].r = opt->c_r;
            g_optStats[optIndex].gamma = opt->c_iterGamma;
            g_optStats[optIndex].normG = opt->c_err;
        }
        
        
        if(opt->c_totalIter > g_mnMaxIterHist)
            g_mnMaxIterHist = opt->c_totalIter;
        if(opt->c_iterGamma > g_mnMaxGammaIter)
            g_mnMaxGammaIter = opt->c_iterGamma;
        for(int r = 0; r < NUM_REGULARIZATIONS; r++)
        {
            if(fabs(opt->c_r - REGULARIZATIONS[r]) < 1e-10)
                g_mnHistReg[r]++;
        }
        
        g_mnIterMean = 
            (g_mnNumDualSolves * g_mnIterMean + opt->c_totalIter) / 
            (g_mnNumDualSolves + 1.0);
        g_mnIterGammaMean = 
            (g_mnNumDualSolves * g_mnIterGammaMean + opt->c_iterGamma) / 
            (g_mnNumDualSolves + 1.0);
        g_mnNumDualSolves++;
        
        delete opt;
    }
}



/*
    Updates one time step.
*/
void update(double dt)
{
    static bool firstTime = true;
    static double *uOld;
    static double *flux;
    int momentGridSize;
    Timer timer;
    
    
    // First time setup.
    if(firstTime) {
        firstTime = false;
        
        momentGridSize = (g_gX[3]-g_gX[0]+1) * (g_gY[3]-g_gY[0]+1) * 
            (g_gZ[3]-g_gZ[0]+1) * g_numMoments;
        uOld = (double*)malloc(momentGridSize * sizeof(double));
        flux = (double*)malloc(momentGridSize * sizeof(double));
    }
    
    
    // Copy initial grid for use later.
    #pragma omp parallel for
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
    for(int m = 0; m < g_numMoments; m++) {
        uOld[IU(i,j,k,m)] = g_u[IU(i,j,k,m)];
    }}}}
    
    
    // Do two Euler steps.
    for(int i = 0; i < 2; i++) {
        timer.restart();
        communicateBoundaries();
        timer.stop();
        g_updateTimes[g_numUpdates * 8 + i * 4] = timer.getTimeElapsed();
        
        
        g_updateTimes[g_numUpdates * 8 + i * 4 + 1] = 0.0;  // For solveOptimization in case of P_N
        if(g_solver == SOLVER_MN) {
            timer.restart();
            if(g_mnBatched)
                solveBatchOptimization(i);
            else
                solveOptimization(i);
            timer.stop();
            g_updateTimes[g_numUpdates * 8 + i * 4 + 1] = timer.getTimeElapsed();
            
            timer.restart();
            solveFluxKinetic(g_u, flux, g_mnAlpha, g_dx, g_dy, g_dz);
            timer.stop();
        }
        else if(g_pnUseKineticFlux == 0) {
            timer.restart();
            solveFluxMoment(g_u, flux, g_dx, g_dy, g_dz);
            timer.stop();
        }
        else {
            timer.restart();
            solveFluxKinetic(g_u, flux, g_u, g_dx, g_dy, g_dz);
            timer.stop();
        }
        g_updateTimes[g_numUpdates * 8 + i * 4 + 2] = timer.getTimeElapsed();
      
        
        timer.restart();
        eulerStep(g_u, flux, dt);
        timer.stop();
        g_updateTimes[g_numUpdates * 8 + i * 4 + 3] = timer.getTimeElapsed();
        
        
        if(g_node == 0) {
            printf("   (%d) Comm: %.5lfs   Opt: %.5lfs   Flux: %.5lfs   Euler: %.5lfs\n", i, 
                   g_updateTimes[g_numUpdates * 8 + i * 4], 
                   g_updateTimes[g_numUpdates * 8 + i * 4 + 1],
                   g_updateTimes[g_numUpdates * 8 + i * 4 + 2],
                   g_updateTimes[g_numUpdates * 8 + i * 4 + 3]);
        }
    }
    g_numUpdates++;
    
    
    // Average initial with second Euler step.
    #pragma omp parallel for
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
    for(int m = 0; m < g_numMoments; m++) {
        g_u[IU(i,j,k,m)] = 0.5 * (g_u[IU(i,j,k,m)] + uOld[IU(i,j,k,m)]);
    }}}}
}
