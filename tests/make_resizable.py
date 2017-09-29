#! /usr/bin/env python
""" Create a HDF5 file with data with an resizable datasets. """

import h5py
import numpy as np

f = h5py.File('resizable.hdf5', 'w', libver='earliest')

f.create_dataset(
    'dataset1', shape=(4, 6), maxshape=(8, 12), dtype='<f8',
    data=np.arange(4 * 6).reshape(4, 6), track_times=False)

# datasets with unlimited dimensions
f.create_dataset(
    'dataset2', shape=(10, 5), maxshape=(10, None), dtype='<i4',
    data=np.arange(10*5).reshape(10, 5), track_times=False)

f.create_dataset(
    'dataset3', shape=(8, 4), maxshape=(None, None), dtype='>i2',
    data=np.arange(8*4).reshape(8, 4), track_times=False)

f.close()
