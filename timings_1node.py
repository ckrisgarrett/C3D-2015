import numpy
import matplotlib.pyplot as plt
import math
import sys

# Get the data from the master file.
filename = sys.argv[1]
f = open(filename, 'r')

numUpdates = int(f.readline())

nodesX = int(f.readline())
nodesY = int(f.readline())
nodesZ = int(f.readline())

filenames = [''] * nodesX*nodesY*nodesZ
for i in range(0, nodesX*nodesY*nodesZ):
    filenames[i] = 'output/' + f.readline().rstrip('\n')

print '--- Data from master file ---'
print 'numUpdates ', str(numUpdates)
print 'nodesX ', str(nodesX)
print 'nodesY ', str(nodesY)
print 'nodesZ ', str(nodesZ)


filename = sys.argv[2]


data = numpy.fromfile(filename)
data.dtype = numpy.double

comm1  = data[0:numUpdates*8:8]
opt1   = data[1:numUpdates*8:8]
flux1  = data[2:numUpdates*8:8]
euler1 = data[3:numUpdates*8:8]
comm2  = data[4:numUpdates*8:8]
opt2   = data[5:numUpdates*8:8]
flux2  = data[6:numUpdates*8:8]
euler2 = data[7:numUpdates*8:8]

print len(comm1)
print len(range(0,numUpdates))



# Plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.plot(range(0,numUpdates), comm1, range(0,numUpdates), comm2)
ax1.set_title('Comm1')
ax2.plot(range(0,numUpdates), opt1, range(0,numUpdates), opt2)
ax2.set_title('opt1')
ax3.plot(range(0,numUpdates), flux1, range(0,numUpdates), flux2)
ax3.set_title('flux1')
ax4.plot(range(0,numUpdates), euler1, range(0,numUpdates), euler2)
ax4.set_title('euler1')
plt.tight_layout()

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
#ax1.plot(range(0,numUpdates), comm2)
#ax1.set_title('Comm2')
#ax2.plot(range(0,numUpdates), opt2)
#ax2.set_title('opt2')
#ax3.plot(range(0,numUpdates), flux2)
#ax3.set_title('flux2')
#ax4.plot(range(0,numUpdates), euler2)
#ax4.set_title('euler2')
#plt.tight_layout()


# Show plots
plt.show()
