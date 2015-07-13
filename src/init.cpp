/*
    File:   init.cpp
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#include "input_deck_reader.h"
#include "global.h"
#include "opt/opt.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <gsl/gsl_linalg.h>

#ifdef USE_MPI
#include <mpi.h>
#endif


/*
    The initial conditions.
*/
static
double setGaussianInitCond(double sigma, double x, double y, double z)
{
    double s2 = sigma * sigma;
    double factor1d = 1.0 / (sigma * sqrt(2.0 * M_PI));
    double factor3d = factor1d * factor1d * factor1d;
    
    return factor3d * exp(-(x * x + y * y + z * z) / (2.0 * s2));
}

static 
double setSmoothInitCond(double x, double y, double z)
{
    return 2.0 + cos(2*M_PI*x) * cos(2*M_PI*y) * cos(2*M_PI*z);
}

static 
double setFunkyInitCond(double sigma, double x, double y, double z)
{
    double retValue = 0.0;
    int i, j, k;
    
    double ax = g_globalAx;
    double ay = g_globalAy;
    double az = g_globalAz;
    double wx = g_globalBx - g_globalAx;
    double wy = g_globalBy - g_globalAy;
    double wz = g_globalBz - g_globalAz;
    
    // Gaussians
    for(i = 1; i < 10; i += 2) {
    for(j = 1; j < 10; j += 2) {
    for(k = 1; k < 10; k += 2) {
        retValue += setGaussianInitCond(sigma, 
                                        x - (ax + i * wx / 10.0), 
                                        y - (ay + j * wy / 10.0), 
                                        z - (az + k * wz / 10.0));
    }}}
    
    // Gaussians plus smooth.
    return setSmoothInitCond(x, y, z) + retValue;
}



/*
    Checks the status of input parameters.
*/
static
void checkInput(bool isOk, int lineNumber)
{
    if(!isOk) {
        printf("init.cpp: input error at line %d\n", lineNumber);
        utils_abort();
    }
}


/*
    Sets up the local computation grid.
*/
void setupLocalMpiInfo(InputDeckReader *inputDeckReader)
{
    // Read data from input deck.
    checkInput(inputDeckReader->getValue("NX", &g_globalNx), __LINE__);
    checkInput(inputDeckReader->getValue("NY", &g_globalNy), __LINE__);
    checkInput(inputDeckReader->getValue("NZ", &g_globalNz), __LINE__);
    checkInput(inputDeckReader->getValue("MPIX", &g_mpix), __LINE__);
    checkInput(inputDeckReader->getValue("MPIY", &g_mpiy), __LINE__);
    checkInput(inputDeckReader->getValue("MPIZ", &g_mpiz), __LINE__);
    checkInput(inputDeckReader->getValue("AX", &g_globalAx), __LINE__);
    checkInput(inputDeckReader->getValue("AY", &g_globalAy), __LINE__);
    checkInput(inputDeckReader->getValue("AZ", &g_globalAz), __LINE__);
    checkInput(inputDeckReader->getValue("BX", &g_globalBx), __LINE__);
    checkInput(inputDeckReader->getValue("BY", &g_globalBy), __LINE__);
    checkInput(inputDeckReader->getValue("BZ", &g_globalBz), __LINE__);
    
    
    // Set dx, dy, dz
    g_dx = (g_globalBx - g_globalAx) / g_globalNx;
    g_dy = (g_globalBy - g_globalAy) / g_globalNy;
    g_dz = (g_globalBz - g_globalAz) / g_globalNz;
    
    
    // For non-MPI mode.
    g_node = 0;
    g_numNodes = 1;
    g_nx = g_globalNx;
    g_ny = g_globalNy;
    g_nz = g_globalNz;
    g_ax = g_globalAx;
    g_ay = g_globalAy;
    g_az = g_globalAz;
    g_bx = g_globalBx;
    g_by = g_globalBy;
    g_bz = g_globalBz;
    
    
    // For MPI mode.
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &g_node);
    MPI_Comm_size(MPI_COMM_WORLD, &g_numNodes);
    
    if((g_globalNx < g_mpix || g_globalNy < g_mpiy || g_globalNz < g_mpiz) && g_node == 0) {
        printf("Init: Number of partitions larger than number of cells.\n");
        utils_abort();
    }
    if(g_mpix * g_mpiy * g_mpiz != g_numNodes && g_node == 0) {
        printf("Init: Number of Nodes (%d) != mpix * mpiy * mpiz.\n", g_numNodes);
        utils_abort();
    }
    g_nodeX = g_node / (g_mpiy * g_mpiz);
    g_nodeY = (g_node / g_mpiz) % g_mpiy;
    g_nodeZ = g_node % g_mpiz;

    if(g_nodeX < g_globalNx % g_mpix) {
        g_nx = g_globalNx / g_mpix + 1;
        g_ax = g_globalAx + g_dx * g_nx * g_nodeX;
        g_bx = g_ax + g_dx * g_nx;
    }
    else {
        g_nx = g_globalNx / g_mpix;
        g_ax = g_globalBx - g_dx * g_nx * (g_mpix - g_nodeX);
        g_bx = g_ax + g_dx * g_nx;
    }
    if(g_nodeY < g_globalNy % g_mpiy) {
        g_ny = g_globalNy / g_mpiy + 1;
        g_ay = g_globalAy + g_dy * g_ny * g_nodeY;
        g_by = g_ay + g_dy * g_ny;
    }
    else {
        g_ny = g_globalNy / g_mpiy;
        g_ay = g_globalBy - g_dy * g_ny * (g_mpiy - g_nodeY);
        g_by = g_ay + g_dy * g_ny;
    }
    if(g_nodeZ < g_globalNz % g_mpiz) {
        g_nz = g_globalNz / g_mpiz + 1;
        g_az = g_globalAz + g_dz * g_nz * g_nodeZ;
        g_bz = g_az + g_dz * g_nz;
    }
    else {
        g_nz = g_globalNz / g_mpiz;
        g_az = g_globalBz - g_dz * g_nz * (g_mpiz - g_nodeZ);
        g_bz = g_az + g_dz * g_nz;
    }
    
    // For debugging.
//     printf("g_numNodes: %d   (%d)\n", g_numNodes, g_node);
//     printf("g_nx: %d   (%d)\n", g_nx, g_node);
//     printf("g_ny: %d   (%d)\n", g_ny, g_node);
//     printf("g_nz: %d   (%d)\n", g_nz, g_node);
//     printf("g_nodeX: %d   (%d)\n", g_nodeX, g_node);
//     printf("g_nodeY: %d   (%d)\n", g_nodeY, g_node);
//     printf("g_nodeZ: %d   (%d)\n", g_nodeZ, g_node);
//     printf("g_ax: %f   (%d)\n", g_ax, g_node);
//     printf("g_ay: %f   (%d)\n", g_ay, g_node);
//     printf("g_az: %f   (%d)\n", g_az, g_node);
//     printf("g_bx: %f   (%d)\n", g_bx, g_node);
//     printf("g_by: %f   (%d)\n", g_by, g_node);
//     printf("g_bz: %f   (%d)\n", g_bz, g_node);
    #endif
}


/*
    Sets up Pn specific things.
*/
static void initPn(InputDeckReader *inputDeckReader, double cflFactor)
{
    checkInput(inputDeckReader->getValue("PN_USE_KINETIC_FLUX", &g_pnUseKineticFlux), __LINE__);
    
    
    // Set up matrices for Pn flux operators.
    for(int i = 0; i < g_numMoments; i++) {
    for(int j = 0; j < g_numMoments; j++) {
        g_pnFluxOperatorXiPlus[IFLUX(i,j)] = 0.0;
        g_pnFluxOperatorXiMinus[IFLUX(i,j)] = 0.0;
        g_pnFluxOperatorEtaPlus[IFLUX(i,j)] = 0.0;
        g_pnFluxOperatorEtaMinus[IFLUX(i,j)] = 0.0;
        g_pnFluxOperatorMuPlus[IFLUX(i,j)] = 0.0;
        g_pnFluxOperatorMuMinus[IFLUX(i,j)] = 0.0;
        
        for(int q = 0; q < g_numQuadPoints; q++) {
            if(g_xi[q] > 0) {
                g_pnFluxOperatorXiPlus[IFLUX(i,j)] += 
                    g_xi[q] * g_m[IM(q,i)] * g_m[IM(q,j)] * g_w[q];
            }
            else {
                g_pnFluxOperatorXiMinus[IFLUX(i,j)] += 
                    g_xi[q] * g_m[IM(q,i)] * g_m[IM(q,j)] * g_w[q];
            }
            if(g_eta[q] > 0) {
                g_pnFluxOperatorEtaPlus[IFLUX(i,j)] += 
                    g_eta[q] * g_m[IM(q,i)] * g_m[IM(q,j)] * g_w[q];
            }
            else {
                g_pnFluxOperatorEtaMinus[IFLUX(i,j)] += 
                    g_eta[q] * g_m[IM(q,i)] * g_m[IM(q,j)] * g_w[q];
            }
            if(g_mu[q] > 0) {
                g_pnFluxOperatorMuPlus[IFLUX(i,j)] += 
                    g_mu[q] * g_m[IM(q,i)] * g_m[IM(q,j)] * g_w[q];
            }
            else {
                g_pnFluxOperatorMuMinus[IFLUX(i,j)] += 
                    g_mu[q] * g_m[IM(q,i)] * g_m[IM(q,j)] * g_w[q];
            }
        }
    }}
    
    // Maximum value for delta t.
    g_maxDt = cflFactor * 0.5 * MIN(MIN(g_dx, g_dy), g_dz);
}


/*
    Sets up Mn specific things.
*/
static void initMn(InputDeckReader *inputDeckReader, double cflFactor)
{
    int numXCells = (g_gX[3] - g_gX[0] + 1);
    int numYCells = (g_gY[3] - g_gY[0] + 1);
    int numZCells = (g_gZ[3] - g_gZ[0] + 1);
    int nnz = 0;  // Number of nonzeros of beta.
    int ptr = 0;  // Index used to go through nonzeros of beta.
    
    checkInput(inputDeckReader->getValue("MN_TOL", &g_mnTol), __LINE__);
    checkInput(inputDeckReader->getValue("MN_COND_H_MAX", &g_mnCondHMax), __LINE__);
    checkInput(inputDeckReader->getValue("MN_MAX_ITER", &g_mnMaxIter), __LINE__);
    checkInput(inputDeckReader->getValue("MN_BATCHED", &g_mnBatched), __LINE__);
    checkInput(inputDeckReader->getValue("MN_TOL_GAMMA", &g_mnTolGamma), __LINE__);
    checkInput(inputDeckReader->getValue("MN_USE_GAUNT_COEF", &g_mnUseGaunt), __LINE__);
    checkInput(inputDeckReader->getValue("MN_USE_GAUNT_SPARSE", &g_mnUseGauntSparse), __LINE__);
    
    g_mnAlpha = (double*)malloc(numXCells * numYCells * numZCells * g_numMoments * sizeof(double));
    
    for(int i = g_gX[0]; i <= g_gX[3]; i++) {
    for(int j = g_gY[0]; j <= g_gY[3]; j++) {
    for(int k = g_gZ[0]; k <= g_gZ[3]; k++) {
        g_mnAlpha[IU(i,j,k,0)] = sqrt(4.0 * M_PI) * log(g_u[IU(i,j,k,0)] / sqrt(4.0 * M_PI));
        for(int m = 1; m < g_numMoments; m++)
            g_mnAlpha[IU(i,j,k,m)] = 0.0;
    }}}
    
    linesearch_init(g_numMoments);
    fobj_init(g_numMoments2);
    
    // Maximum value for delta t.
    g_maxDt = cflFactor * (2.0 / (2.0 + g_theta)) / (1.0/g_dx + 1.0/g_dy + 1.0/g_dz);
    
    
    g_mnMaxIterHist = 0;
    g_mnMaxGammaIter = 0;
    g_mnIterMean = 0.0;
    g_mnIterGammaMean = 0.0;
    g_mnNumDualSolves = 0;
    //g_mnHistReg = new int[NUM_REGULARIZATIONS];  Now in main init section.
    for(int i = 0; i < NUM_REGULARIZATIONS; i++)
        g_mnHistReg[i] = 0;
    
    
    // Gaunt Coefficients
    g_beta = (double*)malloc(g_numMoments * g_numMoments * g_numMoments2 * sizeof(double));
    if(g_mnUseGaunt) {
        gsl_vector *b = gsl_vector_alloc(g_numQuadPoints);
        gsl_vector *beta = gsl_vector_alloc(g_numMoments2);
        gsl_vector *tau = gsl_vector_alloc(g_numMoments2);
        gsl_vector *residual = gsl_vector_alloc(g_numQuadPoints);
        gsl_matrix *A = gsl_matrix_alloc(g_numQuadPoints, g_numMoments2);
        memcpy(A->data, g_m2, g_numQuadPoints * g_numMoments2 * sizeof(double));
        gsl_linalg_QR_decomp(A, tau);
        
        for(int i = 0; i < g_numMoments; i++) {
        for(int j = 0; j < g_numMoments; j++) {
            for(int q = 0; q < g_numQuadPoints; q++) {
                gsl_vector_set(b, q, g_m2[IM2(q,i)] * g_m2[IM2(q,j)]);
            }
            gsl_linalg_QR_lssolve(A, tau, b, beta, residual);
            for(int k = 0; k < g_numMoments2; k++) {
                g_beta[(i*g_numMoments+j)*g_numMoments2+k] = gsl_vector_get(beta, k);
                if(fabs(gsl_vector_get(beta, k)) > 1e-13)
                    nnz++;
            }
        }}
        
        gsl_vector_free(b);
        gsl_vector_free(beta);
        gsl_vector_free(tau);
        gsl_vector_free(residual);
        gsl_matrix_free(A);
        
        
        // Sparse Gaunt Coefficients
        g_betaVal = (double*)malloc(nnz * sizeof(double));
        g_betaColInd = (int*)malloc(nnz * sizeof(int));
        g_betaRowPtr = (int*)malloc((g_numMoments * g_numMoments + 1) * sizeof(int));
        g_betaRowPtr[g_numMoments * g_numMoments] = nnz;
        ptr = 0;
        for(int i = 0; i < g_numMoments; i++) {
        for(int j = 0; j < g_numMoments; j++) {
            int index = i * g_numMoments + j;
            g_betaRowPtr[index] = ptr;
            for(int k = 0; k < g_numMoments2; k++) {
                double val = g_beta[index * g_numMoments2 + k];
                if(fabs(val) > 1e-13) {
                    g_betaVal[ptr] = val;
                    g_betaColInd[ptr] = k;
                    ptr++;
                }
            }
        }}
    }
}


/*
    This function initializes the data.
*/
void init()
{
    int momentOrder, quadOrder;
    double cflFactor;
    double *w;
    double *mu;
    double *phi;
    int numXCells, numYCells, numZCells;
    double crossSection;
    double initFloor;
    double x_i, y_j, z_k;
    double gaussianSigma;
    char initCond[100];
    char solverType[100];
    InputDeckReader inputDeckReader;
    
    
    // Read input.deck
    inputDeckReader.readInputDeck("input.deck");
    
    checkInput(inputDeckReader.getValue("SOLVER", solverType), __LINE__);
    checkInput(inputDeckReader.getValue("TFINAL", &g_tFinal), __LINE__);
    checkInput(inputDeckReader.getValue("OUTDT", &g_outDt), __LINE__);
    checkInput(inputDeckReader.getValue("OUTPUT_DATA", &g_outputData), __LINE__);
    checkInput(inputDeckReader.getValue("OUTPUT_OPT_STATS", &g_outputOptStats), __LINE__);
    checkInput(inputDeckReader.getValue("OUTPUT_TIMINGS", &g_outputTimings), __LINE__);
    checkInput(inputDeckReader.getValue("MOMENT_ORDER", &momentOrder), __LINE__);
    checkInput(inputDeckReader.getValue("QUAD_ORDER", &quadOrder), __LINE__);
    checkInput(inputDeckReader.getValue("CFL_FACTOR", &cflFactor), __LINE__);
    checkInput(inputDeckReader.getValue("CROSS_SECTION", &crossSection), __LINE__);
    checkInput(inputDeckReader.getValue("INIT_FLOOR", &initFloor), __LINE__);
    checkInput(inputDeckReader.getValue("GAUSS_SIGMA", &gaussianSigma), __LINE__);
    checkInput(inputDeckReader.getValue("INIT_COND", initCond), __LINE__);
    checkInput(inputDeckReader.getValue("OMP_THREADS", &g_numOmpThreads), __LINE__);
    checkInput(inputDeckReader.getValue("CUDA_KINETIC", &g_cudaKinetic), __LINE__);
    checkInput(inputDeckReader.getValue("CUDA_BATCHED", &g_cudaBatched), __LINE__);
    checkInput(inputDeckReader.getValue("CUDA_BATCH_SIZE", &g_cudaBatchSize), __LINE__);
    checkInput(inputDeckReader.getValue("THETA", &g_theta), __LINE__);
    
    
    // Setup solver type.
    if(strcmp(solverType, "pn") == 0)
        g_solver = SOLVER_PN;
    else if(strcmp(solverType, "mn") == 0)
        g_solver = SOLVER_MN;
    else {
        printf("Solver type does not exist\n.");
        utils_abort();
    }
    
    // Set initial condition.
    if(strcmp(initCond, "gaussian") == 0)
        g_initCond = INIT_COND_GAUSSIAN;
    else if(strcmp(initCond, "smooth") == 0)
        g_initCond = INIT_COND_SMOOTH;
    else if(strcmp(initCond, "funky") == 0)
        g_initCond = INIT_COND_FUNKY;
    else {
        printf("Initial condition not supported\n.");
        utils_abort();
    }
    
    
    // Setup MPI and OpenMP
    setupLocalMpiInfo(&inputDeckReader);
    
    #ifdef USE_OPENMP
    omp_set_num_threads(g_numOmpThreads);
    #else
    g_numOmpThreads = 1;
    #endif
    
    
    // Print input deck
    if(g_node == 0)
        inputDeckReader.print();
    
    
    // Number of quadrature points and number of moments.
    g_numQuadPoints = 2 * quadOrder * quadOrder;
    g_numMoments    = (momentOrder + 1) * (momentOrder + 1);
    g_numMoments2   = (2 * momentOrder + 1) * (2 * momentOrder + 1);
    
    
    // The weights and nodes for integration and the angles.
    w   = (double*)malloc(quadOrder * sizeof(double));
    mu  = (double*)malloc(quadOrder * sizeof(double));
    phi = (double*)malloc(2 * quadOrder * sizeof(double));
    
    
    // Allocate memory for quadrature.
    g_m  = (double*)malloc(g_numQuadPoints * g_numMoments * sizeof(double));
    g_m2 = (double*)malloc(g_numQuadPoints * g_numMoments2 * sizeof(double));
    g_w  = (double*)malloc(g_numQuadPoints * sizeof(double));
    g_xi  = (double*)malloc(g_numQuadPoints * sizeof(double));
    g_eta = (double*)malloc(g_numQuadPoints * sizeof(double));
    g_mu  = (double*)malloc(g_numQuadPoints * sizeof(double));
    
    
    // Allocate memory for flux operators
    g_pnFluxOperatorXiPlus   = (double*)malloc(g_numMoments * g_numMoments * sizeof(double));
    g_pnFluxOperatorXiMinus  = (double*)malloc(g_numMoments * g_numMoments * sizeof(double));
    g_pnFluxOperatorEtaPlus  = (double*)malloc(g_numMoments * g_numMoments * sizeof(double));
    g_pnFluxOperatorEtaMinus = (double*)malloc(g_numMoments * g_numMoments * sizeof(double));
    g_pnFluxOperatorMuPlus   = (double*)malloc(g_numMoments * g_numMoments * sizeof(double));
    g_pnFluxOperatorMuMinus  = (double*)malloc(g_numMoments * g_numMoments * sizeof(double));
    
    
    // Ghost cell indices
    g_gX[0] = 0;
    g_gX[1] = NUM_GHOST_CELLS;
    g_gX[2] = NUM_GHOST_CELLS + g_nx - 1;
    g_gX[3] = 2 * NUM_GHOST_CELLS + g_nx - 1;
    g_gY[0] = 0;
    g_gY[1] = NUM_GHOST_CELLS;
    g_gY[2] = NUM_GHOST_CELLS + g_ny - 1;
    g_gY[3] = 2 * NUM_GHOST_CELLS + g_ny - 1;
    g_gZ[0] = 0;
    g_gZ[1] = NUM_GHOST_CELLS;
    g_gZ[2] = NUM_GHOST_CELLS + g_nz - 1;
    g_gZ[3] = 2 * NUM_GHOST_CELLS + g_nz - 1;
    
    
    // Allocate memory for grid cells.
    numXCells = (g_gX[3] - g_gX[0] + 1);
    numYCells = (g_gY[3] - g_gY[0] + 1);
    numZCells = (g_gZ[3] - g_gZ[0] + 1);
    g_u = (double*)malloc(numXCells * numYCells * numZCells * g_numMoments * sizeof(double));
    
    
    // Allocate memory for cross sections.
    g_sigmaT = (double*)malloc(numXCells * numYCells * numZCells * sizeof(double));
    g_sigmaS = (double*)malloc(numXCells * numYCells * numZCells * sizeof(double));
    
    
    // Initial Condition
    for(int i = g_gX[0]; i <= g_gX[3]; i++) {
    for(int j = g_gY[0]; j <= g_gY[3]; j++) {
    for(int k = g_gZ[0]; k <= g_gZ[3]; k++) {
        x_i = g_ax + (i-g_gX[1]) * g_dx + g_dx / 2.0;
        y_j = g_ay + (j-g_gY[1]) * g_dy + g_dy / 2.0;
        z_k = g_az + (k-g_gZ[1]) * g_dz + g_dz / 2.0;
        
        g_sigmaT[I3D(i,j,k)] = crossSection;
        g_sigmaS[I3D(i,j,k)] = crossSection;
        
        for(int m = 1; m < g_numMoments; m++)
            g_u[IU(i,j,k,m)] = 0.0;
        
        if(g_initCond == INIT_COND_GAUSSIAN) 
            g_u[IU(i,j,k,0)] = MAX(initFloor, 
                setGaussianInitCond(gaussianSigma, x_i, y_j, z_k) * 2.0 * sqrt(M_PI));
        
        else if(g_initCond == INIT_COND_SMOOTH) {
            g_u[IU(i,j,k,0)] = setSmoothInitCond(x_i, y_j, z_k) * 2.0 * sqrt(M_PI);
            g_u[IU(i,j,k,1)] = 0.3 * g_u[IU(i,j,k,0)];
            g_u[IU(i,j,k,2)] = 0.3 * g_u[IU(i,j,k,0)];
            g_u[IU(i,j,k,3)] = 0.3 * g_u[IU(i,j,k,0)];
        }
        
        else if(g_initCond == INIT_COND_FUNKY)
            g_u[IU(i,j,k,0)] = setFunkyInitCond(gaussianSigma, x_i, y_j, z_k) * 2.0 * sqrt(M_PI);
    }}}
    
    
    // Get quadrature and angles.
    utils_getGaussianWeightsAndNodes(quadOrder, w, mu);
    
    for(int k = 0; k < 2 * quadOrder; k++)
        phi[k] = (k + 0.5) * M_PI / quadOrder;
    
    
    // Setup quadrature.
    for(int q1 = 0; q1 < quadOrder; q1++) {
    for(int q2 = 0; q2 < 2 * quadOrder; q2++) {
        int k = 0;                          // moment counter
        int q = 2 * q1 * quadOrder + q2;    // quadrature counter
        g_w[q] = M_PI / quadOrder * w[q1];
        g_xi[q] = sqrt(1.0 - mu[q1] * mu[q1]) * cos(phi[q2]);
        g_eta[q] = sqrt(1.0 - mu[q1] * mu[q1]) * sin(phi[q2]);
        g_mu[q] = mu[q1];
        
        for(int n = 0; n < momentOrder + 1; n++) {
        for(int m = -n; m <= n; m++) {
            g_m[IM(q,k)] = utils_getSphericalHarmonic(m, n, mu[q1], phi[q2]);
            k++;
        }}
    }}
    
    // For extended version of g_m
    for(int q1 = 0; q1 < quadOrder; q1++) {
    for(int q2 = 0; q2 < 2 * quadOrder; q2++) {
        int k = 0;                          // moment counter
        int q = 2 * q1 * quadOrder + q2;    // quadrature counter
        g_w[q] = M_PI / quadOrder * w[q1];
        g_xi[q] = sqrt(1.0 - mu[q1] * mu[q1]) * cos(phi[q2]);
        g_eta[q] = sqrt(1.0 - mu[q1] * mu[q1]) * sin(phi[q2]);
        g_mu[q] = mu[q1];
        
        for(int n = 0; n < 2 * momentOrder + 1; n++) {
        for(int m = -n; m <= n; m++) {
            g_m2[IM2(q,k)] = utils_getSphericalHarmonic(m, n, mu[q1], phi[q2]);
            k++;
        }}
    }}
    
    
    g_mnHistReg = new int[NUM_REGULARIZATIONS];
    if(g_solver == SOLVER_PN)
        initPn(&inputDeckReader, cflFactor);
    else
        initMn(&inputDeckReader, cflFactor);
    
    
    // Initialize g_updateTimes.
    g_numUpdates = 0;
    g_maxNumUpdates = ceil(g_tFinal / g_maxDt) + 1;
    g_updateTimes = (double*)malloc(8 * g_maxNumUpdates * sizeof(double));
    memset(g_updateTimes, 0, 8 * g_maxNumUpdates * sizeof(double));
    
    
    // Allocate memory for optimization statistics
    if(g_outputOptStats) {
        g_optStats = (OPTIMIZATION_STATS*)malloc(2 * g_maxNumUpdates * g_nx * sizeof(OPTIMIZATION_STATS));
        memset(g_optStats, 0, 2 * g_maxNumUpdates * g_nx * sizeof(OPTIMIZATION_STATS));
    }
    
    
    // Free memory.
    free(w);
    free(mu);
    free(phi);
}

