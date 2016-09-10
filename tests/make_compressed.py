#! /usr/bin/env python
""" Create a HDF5 file with a compressed dataset. """

import h5py
import numpy as np

f = h5py.File('compressed.hdf5', 'w', libver='earliest')

# gzip compressed dataset
f.create_dataset(
    'dataset1', shape=(21, 16), chunks=(2, 2), dtype='<u2',
    compression='gzip', shuffle=False,
    data=np.arange(21*16).reshape(21, 16), track_times=False)

# gzip + shuffled dataset
f.create_dataset(
    'dataset2', shape=(21, 16), chunks=(4, 4), dtype='<i4',
    compression='gzip', shuffle=True,
    data=np.arange(21*16).reshape(21, 16), track_times=False)

# shuffled dataset
f.create_dataset(
    'dataset3', shape=(21, 16), chunks=(7, 4), dtype='<f8',
    compression=None, shuffle=True,
    data=np.arange(21*16).reshape(21, 16), track_times=False)

f.close()
