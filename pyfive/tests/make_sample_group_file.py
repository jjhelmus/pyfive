#! /usr/bin/env python

""" Create a basic HDF5 example with a single array and two attributes. """
import h5py
import numpy as np

f = h5py.File('group_example.hdf5', 'w')

# root group attribute
f.attrs['alice'] = 12

# dataset with two attributes
dset = f.create_dataset('example', (100, ), dtype='i')
dset[...] = np.arange(100)
dset.attrs['foo'] = 99.5
dset.attrs['bar'] = 42

grp = f.create_group('subgroup_one')
grp.attrs['bob'] = 13

dset = grp.create_dataset('grp_one_array', (10, ), dtype='f')
dset[...] = np.arange(10) + 55
dset.attrs['carol'] = 14


f.close()
