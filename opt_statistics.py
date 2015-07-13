import numpy
import matplotlib.pyplot as plt
import math
import sys


filename = sys.argv[1]
data = numpy.fromfile(filename)

data.dtype = numpy.int64
numUpdates = data[0]
nx = data[1]
print 'numUpdates: ', numUpdates
print 'nx:         ', nx

data.dtype = numpy.int64
iters = data[3:len(data):5]

#for i in range(0,2*numUpdates):
    #for j in range(0,nx):
        #print iters[j + i * nx], 
    #print ''
iters = iters.reshape((2*numUpdates,nx))

#numpy.set_printoptions(threshold='nan')
print iters

#cmap = plt.get_cmap('jet', 3)
plt.imshow(iters, origin='lower', interpolation="none")#, cmap=cmap)
plt.colorbar()


# Show plots
plt.show()
