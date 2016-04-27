#! /usr/bin/env python
""" Create a HDF5 file with objects using the earliest library version. """

import h5py
import numpy as np

f = h5py.File('earliest.hdf5', 'w', libver='earliest')

# root group
f.attrs.create('attr1', -123, dtype='<i4')
dset1 = f.create_dataset(
    'dataset1', shape=(4, ), dtype='<i4', data=np.arange(4), track_times=False)
dset1.attrs.create('attr2', 130, dtype='>u1')

# group
grp = f.create_group('group1')
grp.attrs.create('attr3', 12.34, dtype='<f4')
dset2 = grp.create_dataset(
    'dataset2', shape=(4, ), dtype='>u8', data=np.arange(4), track_times=False)
dset2.attrs.create('attr4', b'Hi', dtype='|S2')

# sub-group
subgroup = grp.create_group('subgroup1')
subgroup.attrs['attr5'] = b'Test'
dset3 = subgroup.create_dataset(
    'dataset3', shape=(4, ), dtype='<f4', data=np.arange(4), track_times=False)
dset3.attrs['attr6'] = u'Test' + chr(0x00A7)

f.close()
