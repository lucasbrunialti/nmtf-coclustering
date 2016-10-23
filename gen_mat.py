
import h5py
import numpy as np

X = np.random.rand(20, 15).astype(np.float64)

print X

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.close()

# np.savetxt('data.csv', X, delimiter=',')
