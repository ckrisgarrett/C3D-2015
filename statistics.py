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
print '    sizeX ', str(sizeX)
print '    sizeY ', str(sizeY)
print '    sizeZ ', str(sizeZ)
print '    ax ', str(ax)
print '    ay ', str(ay)
print '    az ', str(az)
print '    bx ', str(bx)
print '    by ', str(by)
print '    bz ', str(bz)
print '    numMoments ', str(numMoments)
print '    nodesX ', str(nodesX)
print '    nodesY ', str(nodesY)
print '    nodesZ ', str(nodesZ)
print ''



numRegularization = 5
numUpdateTimes = 10
for fileIndex in range(0,len(filenames)):
    data = numpy.fromfile(filenames[fileIndex])
    data.dtype = numpy.int64
    nx = data[0]
    ny = data[1]
    nz = data[2]

    print 'Node: ', str(fileIndex)
    offset = nx*ny*nz*numMoments+3
    for i in range(0,numRegularization):
        print '    reg[', i, ']: ', data[offset + i]

    offset = offset + numRegularization
    data.dtype = numpy.double

    for i in range(0,numUpdateTimes):
        print '    time[', i, ']: ', data[offset + i]
