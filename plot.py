import numpy
import matplotlib.pyplot as plt
import math
import sys

# Get the data from the master file.
filename = sys.argv[1]
f = open(filename, 'r')

sizeX = int(f.readline())
sizeY = int(f.readline())
sizeZ = int(f.readline())

ax = float(f.readline())
ay = float(f.readline())
az = float(f.readline())
bx = float(f.readline())
by = float(f.readline())
bz = float(f.readline())

numMoments = int(f.readline())

nodesX = int(f.readline())
nodesY = int(f.readline())
nodesZ = int(f.readline())

filenames = [''] * nodesX*nodesY*nodesZ
for i in range(0, nodesX*nodesY*nodesZ):
    filenames[i] = 'output/' + f.readline().rstrip('\n')

print '--- Data from master file ---'
print 'sizeX ', str(sizeX)
print 'sizeY ', str(sizeY)
print 'sizeZ ', str(sizeZ)
print 'ax ', str(ax)
print 'ay ', str(ay)
print 'az ', str(az)
print 'bx ', str(bx)
print 'by ', str(by)
print 'bz ', str(bz)
print 'numMoments ', str(numMoments)
print 'nodesX ', str(nodesX)
print 'nodesY ', str(nodesY)
print 'nodesZ ', str(nodesZ)


M = numpy.zeros((sizeX,sizeY,sizeZ,numMoments))
nx = 0
ny = 0
nz = 0

index1 = 0
for i in range(0, nodesX):
    index2 = 0
    for j in range(0, nodesY):
        index3 = 0
        for k in range (0, nodesZ):
            data = numpy.fromfile(filenames[k + nodesZ * (j + nodesY * i)])
            
            data.dtype = numpy.int64
            nx = data[0]
            ny = data[1]
            nz = data[2]
            
            data.dtype = numpy.double
            M2 = data[3:nx*ny*nz*numMoments+3]
            M2 = M2.reshape((nx,ny,nz,numMoments))
            
            #for i1 in range(0,nx):
                #for j1 in range(0,ny):
                    #for k1 in range(0,nz):
                        #for m in range(0,numMoments):
                            #M[index1+i1, index2+j1, index3+k1, m] = M2[i1,j1,k1,m]
            M[index1:index1+nx, index2:index2+ny, index3:index3+nz, 0:numMoments] = M2
            #for i1 in range(0,nx):
                #for j1 in range(0,ny):
                    #for k1 in range(0,nz):
                        #for m in range(0,numMoments):
                            #M[i1,j1,k1,m] = M2[m + numMoments * (k1 + nz * (j1 + ny * i1))]
            
            index3 = index3 + nz
        index2 = index2 + ny
    index1 = index1 + nx


M3 = numpy.zeros((sizeX,sizeY));
M4 = numpy.zeros((sizeX,sizeZ));
M5 = numpy.zeros((sizeY,sizeZ));

k = sizeZ/2 + 1
for i in range(0,sizeX):
    for j in range(0,sizeY):
        M3[i,j] = M[i,j,k,0] / (2.0 * math.sqrt(math.pi))
j = sizeY/2 + 1
for i in range(0,sizeX):
    for k in range(0,sizeZ):
        M4[i,k] = M[i,j,k,0] / (2.0 * math.sqrt(math.pi))
i = sizeX/2 + 1
for j in range(0,sizeY):
    for k in range(0,sizeZ):
        M5[j,k] = M[i,j,k,0] / (2.0 * math.sqrt(math.pi))

print numpy.sum(M[:,:,:,0]) * (bx - ax) / sizeX * (by - ay) / sizeY * (bz - az) / sizeZ / (2.0 * math.sqrt(math.pi))
print numpy.sum(M[:,:,:,1]) * (bx - ax) / sizeX * (by - ay) / sizeY * (bz - az) / sizeZ / (2.0 * math.sqrt(math.pi))
print numpy.sum(M[:,:,:,2]) * (bx - ax) / sizeX * (by - ay) / sizeY * (bz - az) / sizeZ / (2.0 * math.sqrt(math.pi))
print numpy.sum(M[:,:,:,3]) * (bx - ax) / sizeX * (by - ay) / sizeY * (bz - az) / sizeZ / (2.0 * math.sqrt(math.pi))

# Plot heat map
plt.figure()
plt.imshow(M3, origin='lower',extent=[ax,bx,ay,by],interpolation="none")
plt.colorbar()
plt.title('z=0')
plt.figure()
plt.imshow(M4, origin='lower',extent=[ax,bx,ay,by],interpolation="none")
plt.colorbar()
plt.title('y=0')
plt.figure()
plt.imshow(M5, origin='lower',extent=[ax,bx,ay,by],interpolation="none")
plt.colorbar()
plt.title('x=0')


# Show plots
plt.show()
