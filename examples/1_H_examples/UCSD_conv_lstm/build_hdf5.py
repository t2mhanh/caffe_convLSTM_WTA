import h5py
import numpy

FILENAME = 'seq2.h5'

# Edit the shape here:
T = 20
N = 1 #16

# My sequence indicators
indicators = numpy.ones((T,N))
print indicators.dtype
print indicators.shape

for i in range(0,N):
  indicators[0][i] = 0
print indicators[0:20]
# print indicators[20:30]

# Open an HDF5 file
h5file = h5py.File( FILENAME, 'w')

# Set Sequence indicator
dataset = h5file.create_dataset('sequence', shape=indicators.shape, dtype=indicators.dtype)
dataset[:] = indicators

h5file.close()
