#! /usr/bin/env python
""" Create a HDF5 file with datasets of many different dimensions . """
import h5py
import numpy as np

f = h5py.File('dataset_multidim.hdf5', 'w')

# signed intergers
common_args = {
    'dtype': '<i4',
    'track_times': False,
}
f.create_dataset('a', shape=(2, ), data=np.arange(2), **common_args)
f.create_dataset('b', shape=(2, 3), data=np.arange(6), **common_args)
f.create_dataset('c', shape=(2, 3, 4), data=np.arange(24), **common_args)
f.create_dataset('d', shape=(2, 3, 4, 5), data=np.arange(120), **common_args)
f.close()
