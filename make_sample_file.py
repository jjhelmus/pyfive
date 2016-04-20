#! /usr/bin/env python

""" Create a basic HDF5 example with a single array and two attributes. """
import h5py
import numpy as np

f = h5py.File('basic_example.hdf5', 'w')
dset = f.create_dataset('example', (100, ), dtype='i')
dset[...] = np.arange(100)
dset.attrs['foo'] = 99.5
dset.attrs['bar'] = 42
f.close()
