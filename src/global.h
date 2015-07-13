/*
    File:   global.h
    Author: Kris Garrett
    Date:   September 6, 2013
*/

#ifndef __GLOBAL_H
#define __GLOBAL_H


#include <stdint.h>


// Macros for indexing arrays.
#define I3D(i,j,k) ( (k) + (g_gZ[3] - g_gZ[0] + 1) * ((j) + (g_gY[3] - g_gY[0] + 1) * (i)) )
#define IU(i,j,k,m) ( (m) + (g_numMoments) * ((k) + (g_gZ[3] - g_gZ[0] + 1) * ((j) + (g_gY[3] - g_gY[0] + 1) * (i))) )
#define IFLUX(i,j) ( (j) + (g_numMoments) * (i) )
#define IM(q,m) ( (m) + (g_numMoments) * (q) )
#define IM2(q,m) ( (m) + (g_numMoments2) * (q) )

#define NUM_GHOST_CELLS 2


// A hack so I only have to declare the global variables here.
#ifdef NO_EXTERN
#define EXTERN 
#else
#define EXTERN extern
#endif


// Global function declarations.
void init();
void outputData(double time);
void update(double dt);
void communicateBoundaries();
void solveFluxMoment(double *u, double *flux, double dx, double dy, double dz);
void solveFluxKinetic(double *u, double *flux, double *alpha, double dx, double dy, double dz);
void outputTimings();
void outputOptStats();


// Enum Types
enum SOLVER_TYPE {
    SOLVER_PN, SOLVER_MN
};

enum INIT_COND {
    INIT_COND_GAUSSIAN, INIT_COND_SMOOTH, INIT_COND_FUNKY
};


struct OPTIMIZATION_STATS
{
    double r;
    int64_t iter;
    int64_t iterGamma;
    double gamma;
    double normG;
};



// All the global variables.
EXTERN double g_crossSection;
EXTERN int g_node;
EXTERN int g_numNodes;
EXTERN bool g_outputData;
EXTERN SOLVER_TYPE g_solver;
EXTERN INIT_COND g_initCond;
EXTERN int g_nx, g_ny, g_nz;
EXTERN int g_globalNx, g_globalNy, g_globalNz;
EXTERN int g_mpix, g_mpiy, g_mpiz;
EXTERN double g_ax, g_ay, g_az, g_bx, g_by, g_bz;
EXTERN double g_globalAx, g_globalAy, g_globalAz, 
              g_globalBx, g_globalBy, g_globalBz;
EXTERN int g_numOmpThreads;
EXTERN double g_dx, g_dy, g_dz;
EXTERN int g_nodeX, g_nodeY, g_nodeZ;
EXTERN int g_numQuadPoints;
EXTERN int g_numMoments;
EXTERN int g_numMoments2;
EXTERN int g_pnUseKineticFlux;
EXTERN double *g_m;
EXTERN double *g_m2;
EXTERN double *g_w;
EXTERN double *g_xi;
EXTERN double *g_eta;
EXTERN double *g_mu;
EXTERN int g_gX[4], g_gY[4], g_gZ[4];
EXTERN double *g_u;
EXTERN double *g_sigmaT;
EXTERN double *g_sigmaS;
EXTERN double g_tFinal;
EXTERN double g_outDt;
EXTERN double g_maxDt;
EXTERN double g_theta;

EXTERN int g_maxNumUpdates;
EXTERN int g_numUpdates;
EXTERN double *g_updateTimes;
EXTERN bool g_outputTimings;

EXTERN double *g_pnFluxOperatorXiPlus;
EXTERN double *g_pnFluxOperatorXiMinus;
EXTERN double *g_pnFluxOperatorEtaPlus;
EXTERN double *g_pnFluxOperatorEtaMinus;
EXTERN double *g_pnFluxOperatorMuPlus;
EXTERN double *g_pnFluxOperatorMuMinus;

EXTERN double *g_mnAlpha;
EXTERN int g_mnMaxIter;
EXTERN double g_mnTol;
EXTERN double g_mnCondHMax;
EXTERN bool g_mnUseGaunt;
EXTERN bool g_mnUseGauntSparse;
EXTERN bool g_mnBatched;
EXTERN double g_mnTolGamma;
EXTERN int g_mnMaxIterHist, g_mnMaxGammaIter;
EXTERN int *g_mnHistReg;
EXTERN double g_mnIterMean;
EXTERN double g_mnIterGammaMean;
EXTERN int g_mnNumDualSolves;
EXTERN OPTIMIZATION_STATS *g_optStats;
EXTERN bool g_outputOptStats;

EXTERN bool g_cudaKinetic;
EXTERN bool g_cudaBatched;
EXTERN int g_cudaBatchSize;

#endif
