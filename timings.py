import numpy
import matplotlib.pyplot as plt
import math
import sys

# Get the data from the master file.
filename = sys.argv[1]
f = open(filename, 'r')

numUpdate = int(f.readline())

nodesX = int(f.readline())
nodesY = int(f.readline())
nodesZ = int(f.readline())

filenames = [''] * nodesX*nodesY*nodesZ
for i in range(0, nodesX*nodesY*nodesZ):
    filenames[i] = 'output/' + f.readline().rstrip('\n')

print '--- Data from master file ---'
print 'numUpdates ', str(numMoments)
print 'nodesX ', str(nodesX)
print 'nodesY ', str(nodesY)
print 'nodesZ ', str(nodesZ)


M = numpy.zeros((nodesX, nodesY, nodesZ, numUpdates))

for i in range(0, nodesX):
    for j in range(0, nodesY):
        for k in range (0, nodesZ):
            data = numpy.fromfile(filenames[k + nodesZ * (j + nodesY * i)])
            data.dtype = numpy.double
            M[i, j, k, 0:numUpdates] = data;

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

# Plot heat map
plt.figure()
plt.imshow(M3, origin='lower',extent=[ax,bx,ay,by],interpolation=None)
plt.colorbar()
plt.figure()
plt.imshow(M4, origin='lower',extent=[ax,bx,ay,by],interpolation=None)
plt.colorbar()
plt.figure()
plt.imshow(M5, origin='lower',extent=[ax,bx,ay,by],interpolation=None)
plt.colorbar()


# Show plots
plt.show()
