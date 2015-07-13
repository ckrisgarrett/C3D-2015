/*
    File:   comm.cpp
    Author: Kris Garrett
    Date:   September 6, 2013
*/

#include "global.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>


#ifdef USE_MPI
#include <mpi.h>


/*
    Returns MPI communication sizes in bytes.
*/
static 
double getBoundarySizeX()
{
    return (g_gY[3] - g_gY[0] + 1) * (g_gZ[3] - g_gZ[0] + 1) * 
        NUM_GHOST_CELLS * g_numMoments * sizeof(double);
}

static 
double getBoundarySizeY()
{
    return (g_gX[3] - g_gX[0] + 1) * (g_gZ[3] - g_gZ[0] + 1) * 
        NUM_GHOST_CELLS * g_numMoments * sizeof(double);
}

static 
double getBoundarySizeZ()
{
    return (g_gX[3] - g_gX[0] + 1) * (g_gY[3] - g_gY[0] + 1) * 
        NUM_GHOST_CELLS * g_numMoments * sizeof(double);
}


/*
    Returns the boundaries inside the domain (ie not the ghost cells).
*/
static 
void getInnerBoundaries(char *xPos, char *xNeg, char *yPos, char *yNeg, char *zPos, char *zNeg)
{
    int x1 = g_gX[1];
    int x2 = g_gX[2] + 1 - NUM_GHOST_CELLS;
    int y1 = g_gY[1];
    int y2 = g_gY[2] + 1 - NUM_GHOST_CELLS;
    int z1 = g_gZ[1];
    int z2 = g_gZ[2] + 1 - NUM_GHOST_CELLS;
    int index;
    
    
    // x
    for(int i = 0; i < NUM_GHOST_CELLS; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        index = k + (g_gZ[2] - g_gZ[1] + 1) * (j + i * (g_gY[2] - g_gY[1] + 1));
        memcpy(&xPos[index * g_numMoments * sizeof(double)], &g_u[IU(x2+i,j,k,0)], 
               g_numMoments * sizeof(double));
        memcpy(&xNeg[index * g_numMoments * sizeof(double)], &g_u[IU(x1+i,j,k,0)], 
               g_numMoments * sizeof(double));
    }}}
    
    // y
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = 0; j < NUM_GHOST_CELLS; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        index = k + (g_gZ[2] - g_gZ[1] + 1) * (j + i * NUM_GHOST_CELLS);
        memcpy(&yPos[index * g_numMoments * sizeof(double)], &g_u[IU(i,y2+j,k,0)], 
               g_numMoments * sizeof(double));
        memcpy(&yNeg[index * g_numMoments * sizeof(double)], &g_u[IU(i,y1+j,k,0)], 
               g_numMoments * sizeof(double));
    }}}
    
    // z
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = 0; k < NUM_GHOST_CELLS; k++) {
        index = k + NUM_GHOST_CELLS * (j + i * (g_gY[2] - g_gY[1] + 1));
        memcpy(&zPos[index * g_numMoments * sizeof(double)], &g_u[IU(i,j,z2+k,0)], 
               g_numMoments * sizeof(double));
        memcpy(&zNeg[index * g_numMoments * sizeof(double)], &g_u[IU(i,j,z1+k,0)], 
               g_numMoments * sizeof(double));
    }}}
}


/*
    Puts the input data into the ghost cells.
*/
static 
void setOuterBoundaries(char *xPos, char *xNeg, char *yPos, char *yNeg, char *zPos, char *zNeg)
{
    int x1 = g_gX[0];
    int x2 = g_gX[3] + 1 - NUM_GHOST_CELLS;
    int y1 = g_gY[0];
    int y2 = g_gY[3] + 1 - NUM_GHOST_CELLS;
    int z1 = g_gZ[0];
    int z2 = g_gZ[3] + 1 - NUM_GHOST_CELLS;
    int index;
    
    
    // x
    for(int i = 0; i < NUM_GHOST_CELLS; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        index = k + (g_gZ[2] - g_gZ[1] + 1) * (j + i * (g_gY[2] - g_gY[1] + 1));
        memcpy(&g_u[IU(x2+i,j,k,0)], &xPos[index * g_numMoments * sizeof(double)], 
               g_numMoments * sizeof(double));
        memcpy(&g_u[IU(x1+i,j,k,0)], &xNeg[index * g_numMoments * sizeof(double)], 
               g_numMoments * sizeof(double));
    }}}
    
    // y
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = 0; j < NUM_GHOST_CELLS; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
        index = k + (g_gZ[2] - g_gZ[1] + 1) * (j + i * NUM_GHOST_CELLS);
        memcpy(&g_u[IU(i,y2+j,k,0)], &yPos[index * g_numMoments * sizeof(double)], 
               g_numMoments * sizeof(double));
        memcpy(&g_u[IU(i,y1+j,k,0)], &yNeg[index * g_numMoments * sizeof(double)], 
               g_numMoments * sizeof(double));
    }}}
    
    // z
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = 0; k < NUM_GHOST_CELLS; k++) {
        index = k + NUM_GHOST_CELLS * (j + i * (g_gY[2] - g_gY[1] + 1));
        memcpy(&g_u[IU(i,j,z2+k,0)], &zPos[index * g_numMoments * sizeof(double)], 
               g_numMoments * sizeof(double));
        memcpy(&g_u[IU(i,j,z1+k,0)], &zNeg[index * g_numMoments * sizeof(double)], 
               g_numMoments * sizeof(double));
    }}}
}


/*
    Given the node for the x,y,z direction, returns the 1D index for the node.
*/
static
int mpiIndex(char direction, int node)
{
    int nodeX = g_nodeX;
    int nodeY = g_nodeY;
    int nodeZ = g_nodeZ;
    
    if(direction == 'x')
        nodeX = (node + g_mpix) % g_mpix;
    else if(direction == 'y')
        nodeY = (node + g_mpiy) % g_mpiy;
    else if(direction == 'z')
        nodeZ = (node + g_mpiz) % g_mpiz;
    else {
        printf("Direction MPI Error.\n");
        utils_abort();
    }
    
    return nodeZ + g_mpiz * (nodeY + g_mpiy * nodeX);
}


/*
    MPI version of communicateBoundaries.  It implements periodic boundary conditions.
*/
void communicateBoundaries()
{
    // Variables.
    int boundarySizeX = getBoundarySizeX();
    int boundarySizeY = getBoundarySizeY();
    int boundarySizeZ = getBoundarySizeZ();
    int sendXPosTag = 1;
    int recvXNegTag = 1;
    int sendXNegTag = 2;
    int recvXPosTag = 2;
    int sendYPosTag = 3;
    int recvYNegTag = 3;
    int sendYNegTag = 4;
    int recvYPosTag = 4;
    int sendZPosTag = 5;
    int recvZNegTag = 5;
    int sendZNegTag = 6;
    int recvZPosTag = 6;
    MPI_Request mpiRequest[12];
    int mpiError;
    
    // Variables to allocate only once.
    static bool firstTime = true;
    static char *sendXPos = NULL;
    static char *sendXNeg = NULL;
    static char *recvXPos = NULL;
    static char *recvXNeg = NULL;
    static char *sendYPos = NULL;
    static char *sendYNeg = NULL;
    static char *recvYPos = NULL;
    static char *recvYNeg = NULL;
    static char *sendZPos = NULL;
    static char *sendZNeg = NULL;
    static char *recvZPos = NULL;
    static char *recvZNeg = NULL;
    
    
    // First time initialization.
    if(firstTime) {
        firstTime = false;
        
        sendXPos = (char*)malloc(boundarySizeX);
        sendXNeg = (char*)malloc(boundarySizeX);
        recvXPos = (char*)malloc(boundarySizeX);
        recvXNeg = (char*)malloc(boundarySizeX);
        sendYPos = (char*)malloc(boundarySizeY);
        sendYNeg = (char*)malloc(boundarySizeY);
        recvYPos = (char*)malloc(boundarySizeY);
        recvYNeg = (char*)malloc(boundarySizeY);
        sendZPos = (char*)malloc(boundarySizeZ);
        sendZNeg = (char*)malloc(boundarySizeZ);
        recvZPos = (char*)malloc(boundarySizeZ);
        recvZNeg = (char*)malloc(boundarySizeZ);
        
        if(sendXPos == NULL || sendXNeg == NULL || recvXPos == NULL || recvXNeg == NULL || 
           sendYPos == NULL || sendYNeg == NULL || recvYPos == NULL || recvYNeg == NULL || 
           sendZPos == NULL || sendZNeg == NULL || recvZPos == NULL || recvZNeg == NULL)
        {
            printf("comm.cpp: Memory allocation failed.\n");
            utils_abort();
        }
    }
    

    // Get boundaries to send.
    getInnerBoundaries(sendXPos, sendXNeg, sendYPos, sendYNeg, sendZPos, sendZNeg);
    
    
    // Send
    MPI_Isend(sendXPos, boundarySizeX, MPI_CHAR, mpiIndex('x', g_nodeX + 1), sendXPosTag, 
              MPI_COMM_WORLD, &mpiRequest[0]);
    MPI_Isend(sendXNeg, boundarySizeX, MPI_CHAR, mpiIndex('x', g_nodeX - 1), sendXNegTag, 
              MPI_COMM_WORLD, &mpiRequest[1]);
    MPI_Isend(sendYPos, boundarySizeY, MPI_CHAR, mpiIndex('y', g_nodeY + 1), sendYPosTag, 
              MPI_COMM_WORLD, &mpiRequest[2]);
    MPI_Isend(sendYNeg, boundarySizeY, MPI_CHAR, mpiIndex('y', g_nodeY - 1), sendYNegTag, 
              MPI_COMM_WORLD, &mpiRequest[3]);
    MPI_Isend(sendZPos, boundarySizeZ, MPI_CHAR, mpiIndex('z', g_nodeZ + 1), sendZPosTag, 
              MPI_COMM_WORLD, &mpiRequest[4]);
    MPI_Isend(sendZNeg, boundarySizeZ, MPI_CHAR, mpiIndex('z', g_nodeZ - 1), sendZNegTag, 
              MPI_COMM_WORLD, &mpiRequest[5]);
    
    // Recv
    MPI_Irecv(recvXPos, boundarySizeX, MPI_CHAR, mpiIndex('x', g_nodeX + 1), recvXPosTag, 
              MPI_COMM_WORLD, &mpiRequest[6]);
    MPI_Irecv(recvXNeg, boundarySizeX, MPI_CHAR, mpiIndex('x', g_nodeX - 1), recvXNegTag, 
              MPI_COMM_WORLD, &mpiRequest[7]);
    MPI_Irecv(recvYPos, boundarySizeY, MPI_CHAR, mpiIndex('y', g_nodeY + 1), recvYPosTag, 
              MPI_COMM_WORLD, &mpiRequest[8]);
    MPI_Irecv(recvYNeg, boundarySizeY, MPI_CHAR, mpiIndex('y', g_nodeY - 1), recvYNegTag, 
              MPI_COMM_WORLD, &mpiRequest[9]);
    MPI_Irecv(recvZPos, boundarySizeZ, MPI_CHAR, mpiIndex('z', g_nodeZ + 1), recvZPosTag, 
              MPI_COMM_WORLD, &mpiRequest[10]);
    MPI_Irecv(recvZNeg, boundarySizeZ, MPI_CHAR, mpiIndex('z', g_nodeZ - 1), recvZNegTag, 
              MPI_COMM_WORLD, &mpiRequest[11]);
    
    // Wait for Send/Recv
    mpiError = MPI_Waitall(12, mpiRequest, MPI_STATUSES_IGNORE);
    if(mpiError != MPI_SUCCESS) {
        printf("comm.cpp: MPI Error %d.\n", mpiError);
        utils_abort();
    }
    
    
    // Set ghost boundaries.
    setOuterBoundaries(recvXPos, recvXNeg, recvYPos, recvYNeg, recvZPos, recvZNeg);
}
#else
/*
    Serial version of communicateBoundaries.  It implements periodic boundary conditions.
*/
void communicateBoundaries()
{
    int x0 = g_gX[0];
    int x1 = g_gX[1];
    int x2 = g_gX[2] + 1 - NUM_GHOST_CELLS;
    int x3 = g_gX[3] + 1 - NUM_GHOST_CELLS;
    int y0 = g_gY[0];
    int y1 = g_gY[1];
    int y2 = g_gY[2] + 1 - NUM_GHOST_CELLS;
    int y3 = g_gY[3] + 1 - NUM_GHOST_CELLS;
    int z0 = g_gZ[0];
    int z1 = g_gZ[1];
    int z2 = g_gZ[2] + 1 - NUM_GHOST_CELLS;
    int z3 = g_gZ[3] + 1 - NUM_GHOST_CELLS;
    
    
    // x
    for(int i = 0; i < NUM_GHOST_CELLS; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
    for(int m = 0; m < g_numMoments; m++) {
        g_u[IU(x0+i,j,k,m)] = g_u[IU(x2+i,j,k,m)];
        g_u[IU(x3+i,j,k,m)] = g_u[IU(x1+i,j,k,m)];
    }}}}
    
    // y
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = 0; j < NUM_GHOST_CELLS; j++) {
    for(int k = g_gZ[1]; k <= g_gZ[2]; k++) {
    for(int m = 0; m < g_numMoments; m++) {
        g_u[IU(i,y0+j,k,m)] = g_u[IU(i,y2+j,k,m)];
        g_u[IU(i,y3+j,k,m)] = g_u[IU(i,y1+j,k,m)];
    }}}}
    
    // z
    for(int i = g_gX[1]; i <= g_gX[2]; i++) {
    for(int j = g_gY[1]; j <= g_gY[2]; j++) {
    for(int k = 0; k < NUM_GHOST_CELLS; k++) {
    for(int m = 0; m < g_numMoments; m++) {
        g_u[IU(i,j,z0+k,m)] = g_u[IU(i,j,z2+k,m)];
        g_u[IU(i,j,z3+k,m)] = g_u[IU(i,j,z1+k,m)];
    }}}}
}
#endif

