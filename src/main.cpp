/*
    File:   main.cpp
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#include "utils.h"
#include "timer.h"
#include "global.h"
#include "input_deck_reader.h"
#include "opt/opt.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif


/*
    Start the program.
*/
int main(int argc, char **argv)
{
    // Variables
    Timer timer;
    double dt, t;
    
    
    // Initialize MPI.
    #ifdef USE_MPI
    MPI_Init(&argc, &argv);
    #endif
    
    
    // Initialize data.
    init();
    if(g_node == 0)
        printf("Init done.\n");
    
    
    // Start Timing
    timer.start();
    
    
    // Start loop in time.
    t = 0.0;
    dt = g_maxDt;
    outputData(t);
    for(double tOut = MIN(t + g_outDt, g_tFinal); tOut <= g_tFinal; 
    tOut = MIN(tOut + g_outDt, g_tFinal)) {
        dt = g_maxDt;
        while(t < tOut - 1e-14 * tOut) {
            t += dt;
            if(t > tOut) {
                dt = tOut - (t - dt);
                t = tOut;
            }
            
            update(dt);
            
            if(g_node == 0)
                printf("dt = %f   t = %f\n", dt, t);
        }
        
        outputData(t);
        if(tOut > g_tFinal - 1e-14 * g_tFinal)
            break;
    }
    
    
    // Print time taken by the program.
    #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    if(g_node == 0) {
        timer.stop();
        printf("Time taken: %.3lfs\n", timer.getTimeElapsed());
        
        if(g_solver == SOLVER_MN) {
            printf("g_mnMaxIterHist = %d\n", g_mnMaxIterHist);
            printf("g_mnMaxGammaIter = %d\n", g_mnMaxGammaIter);
            printf("g_mnIterMean = %f\n", g_mnIterMean);
            printf("g_mnIterGammaMean = %f\n", g_mnIterGammaMean);
            printf("g_mnNumDualSolves = %d\n", g_mnNumDualSolves);
            
            for(int i = 0; i < NUM_REGULARIZATIONS; i++)
                printf("  g_mnHistReg[%d] = %d\n", i, g_mnHistReg[i]);
        }
    }
    
    
    // Output timings.
    outputTimings();
    
    // Output optimization statistics.
    outputOptStats();
    
    
    // End MPI.
    #ifdef USE_MPI
    MPI_Finalize();
    #endif
    
    return 0;
}
