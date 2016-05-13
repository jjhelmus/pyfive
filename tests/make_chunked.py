#! /usr/bin/env python
""" Create a HDF5 file with a chunked dataset. """

import h5py
import numpy as np

f = h5py.File('chunked.hdf5', 'w', libver='earliest')

# root group
dset1 = f.create_dataset(
    'dataset1', shape=(21, 16), chunks=(2, 2), dtype='<i4',
    data=np.arange(21*16).reshape(21, 16), track_times=False)
dset1.attrs.create('attr1', 130, dtype='>u1')

f.close()
