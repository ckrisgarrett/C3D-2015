/*
    File:   output.cpp
    Author: Kris Garrett
    Date:   September 3, 2013
*/

#include "global.h"
#include "utils.h"
#include "opt/opt.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>


/*
    Writes raw data to output file.  If this is node 0
    it also writes the master file.
*/
void outputData(double time)
{
    const int FILENAME_SIZE = 256;
    char filename[FILENAME_SIZE];
    FILE *file;
    int64_t temp;
    double tempFloat;

    
    // Don't output data if it isn't required.
    if(g_outputData == false)
        return;
    
    
    // Master file.
    if(g_node == 0) {
        // Create and open file.
        snprintf(filename, FILENAME_SIZE, "output/out_%.6f.master", time);
        file = fopen(filename, "w");
        if(file == NULL) {
            printf("outputData.cpp: Could not open %s.\n", filename);
            utils_abort();
        }
        
        // sizeX, sizeY, sizeZ
        fprintf(file, "%d\n", g_globalNx);
        fprintf(file, "%d\n", g_globalNy);
        fprintf(file, "%d\n", g_globalNz);
        
        // Domain bounds
        fprintf(file, "%f\n", g_globalAx);
        fprintf(file, "%f\n", g_globalAy);
        fprintf(file, "%f\n", g_globalAz);
        fprintf(file, "%f\n", g_globalBx);
        fprintf(file, "%f\n", g_globalBy);
        fprintf(file, "%f\n", g_globalBz);
        
        // number of moments
        fprintf(file, "%d\n", g_numMoments);
        
        // number of nodes in each direction
        #ifdef USE_MPI
            fprintf(file, "%d\n", g_mpix);
            fprintf(file, "%d\n", g_mpiy);
            fprintf(file, "%d\n", g_mpiz);
        #else
            fprintf(file, "1\n");
            fprintf(file, "1\n");
            fprintf(file, "1\n");
        #endif
        
        // filenames
        for(int i = 0; i < g_numNodes; i++) {
            snprintf(filename, FILENAME_SIZE, "out_%.6f_%d.pn", time, i);
            fprintf(file, "%s\n", filename);
        }
        
        // Close file.
        fclose(file);
    }
    
    
    // Create and open file for data.
    snprintf(filename, FILENAME_SIZE, "output/out_%.6f_%d.pn", time, g_node);
    file = fopen(filename, "wb");
    if(file == NULL) {
        printf("outputData.cpp: Could not open %s.\n", filename);
        utils_abort();
    }
    
    // nx, ny, nz
    temp = g_nx;  fwrite(&temp, sizeof(int64_t), 1, file);
    temp = g_ny;  fwrite(&temp, sizeof(int64_t), 1, file);
    temp = g_nz;  fwrite(&temp, sizeof(int64_t), 1, file);
    
    // data
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        fwrite(&g_u[IU(i,j,k,0)], sizeof(double), g_numMoments, file);
    }}}
    
    // output regularization data.
    temp = g_mnMaxIterHist; fwrite(&temp, sizeof(int64_t), 1, file);
    temp = g_mnMaxGammaIter; fwrite(&temp, sizeof(int64_t), 1, file);
    tempFloat = g_mnIterMean; fwrite(&tempFloat, sizeof(double), 1, file);
    tempFloat = g_mnIterGammaMean; fwrite(&tempFloat, sizeof(double), 1, file);
    temp = g_mnNumDualSolves; fwrite(&temp, sizeof(int64_t), 1, file);
    for(int i = 0; i < NUM_REGULARIZATIONS; i++) {
        temp = g_mnHistReg[i];  fwrite(&temp, sizeof(int64_t), 1, file);
    }
    
    // Close file.
    fclose(file);
}



/*
    Writes timing data.
*/
void outputTimings()
{
    const int FILENAME_SIZE = 256;
    char filename[FILENAME_SIZE];
    FILE *file;

    
    // Don't output data if it isn't required.
    if(g_outputTimings == false)
        return;
    
    
    // Master file.
    if(g_node == 0) {
        // Create and open file.
        snprintf(filename, FILENAME_SIZE, "output/times.master");
        file = fopen(filename, "w");
        if(file == NULL) {
            printf("outputData.cpp: Could not open %s.\n", filename);
            utils_abort();
        }
        
        // number of times steps taken
        fprintf(file, "%d\n", g_numUpdates);
        
        // number of nodes in each direction
        #ifdef USE_MPI
            fprintf(file, "%d\n", g_mpix);
            fprintf(file, "%d\n", g_mpiy);
            fprintf(file, "%d\n", g_mpiz);
        #else
            fprintf(file, "1\n");
            fprintf(file, "1\n");
            fprintf(file, "1\n");
        #endif
        
        // filenames
        for(int i = 0; i < g_numNodes; i++) {
            snprintf(filename, FILENAME_SIZE, "times_%d.dat", i);
            fprintf(file, "%s\n", filename);
        }
        
        // Close file.
        fclose(file);
    }
    
    
    // Create and open file for data.
    snprintf(filename, FILENAME_SIZE, "output/times_%d.dat", g_node);
    file = fopen(filename, "wb");
    if(file == NULL) {
        printf("outputData.cpp: Could not open %s.\n", filename);
        utils_abort();
    }
    
    
    // timings
    fwrite(g_updateTimes, sizeof(double), g_numUpdates * 8, file);
    
    
    // Close file.
    fclose(file);
}


/*
    Writes optimizations statistics.
*/
void outputOptStats()
{
    const int FILENAME_SIZE = 256;
    char filename[FILENAME_SIZE];
    FILE *file;
    int64_t temp;
    int tIndex;

    
    // Don't output data if it isn't required.
    if(g_outputOptStats == false)
        return;
    
    
    // Create and open file for data.
    snprintf(filename, FILENAME_SIZE, "output/stats_%d.dat", g_node);
    file = fopen(filename, "wb");
    if(file == NULL) {
        printf("outputData.cpp: Could not open %s.\n", filename);
        utils_abort();
    }
    
    
    // output statistics
    temp = g_numUpdates;  fwrite(&temp, sizeof(int64_t), 1, file);
    temp = g_nx;          fwrite(&temp, sizeof(int64_t), 1, file);
    
    for(tIndex = 0; tIndex < g_numUpdates * 2; tIndex++) {
        fwrite(&g_optStats[tIndex * g_nx], sizeof(OPTIMIZATION_STATS), g_nx, file);
    }
    
    
    // Close file.
    fclose(file);
}

