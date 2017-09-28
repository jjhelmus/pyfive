#! /usr/bin/env python
""" Create two HDF5 files with datasets with and without fillvalues. """
import h5py
import numpy as np


def make_fillvalue(f):

    common_args = {
        'shape': (4, ),
        'data': np.arange(4),
        'track_times': False,
    }

    f.create_dataset('dset1', dtype='<i1', fillvalue=42, **common_args)
    f.create_dataset('dset2', dtype='<i1', **common_args)
    f.create_dataset('dset3', dtype='<f4', fillvalue=99.5, **common_args)


with h5py.File('fillvalue_earliest.hdf5', 'w', libver='earliest') as f:
    make_fillvalue(f)

with h5py.File('fillvalue_latest.hdf5', 'w', libver='latest') as f:
    make_fillvalue(f)
