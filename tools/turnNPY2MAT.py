import numpy as np
import h5py
matrix = np.load('data/bowArray.npy')
print matrix.shape
f = h5py.File('data/bowMatrix.mat', 'w')
f.create_dataset('bow', data=matrix)
