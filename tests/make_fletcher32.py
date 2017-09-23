#! /usr/bin/env python
""" Create a HDF5 file with datasets with fletcher32 checksums. """

import h5py
import numpy as np

f = h5py.File('fletcher32.hdf5', 'w', libver='earliest')

# even sized dataset with multiple chunks
dset1 = f.create_dataset(
    'dataset1', shape=(4, 4), chunks=(2, 2), dtype='<i4',
    data=np.arange(4*4).reshape(4, 4), track_times=False, fletcher32=True)

# odd sized dataset with single chunk
f.create_dataset(
    'dataset2', shape=(3,), dtype='>i1',
    data=np.arange(3), track_times=False, fletcher32=True)

f.close()
