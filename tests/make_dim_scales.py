#! /usr/bin/env python
""" Create a HDF5 file with dimension scales and labels . """
import h5py
import numpy as np

f = h5py.File('dim_scales.hdf5', 'w')

common_args = {
    'track_times': False,
    'dtype': 'i4',
}

# dataset with dimension labels and scales
f.create_dataset('dset1', data=np.ones((4, 3, 2)), **common_args)

# dimension labels
f['dset1'].dims[0].label = 'z'
f['dset1'].dims[1].label = 'y'
f['dset1'].dims[2].label = 'x'

# dimension scales
f.create_dataset('x1', data=[1, 2], **common_args)
f.create_dataset('y1', data=[3, 4, 5], **common_args)
f.create_dataset('z1', data=[0, 10, 20, 30], **common_args)
f.create_dataset('x2', data=[99, 98], **common_args)

f['dset1'].dims.create_scale(f['x1'], 'x1_name')
f['dset1'].dims.create_scale(f['y1'], 'y1_name')
f['dset1'].dims.create_scale(f['z1'], 'z1_name')

f['dset1'].dims[0].attach_scale(f['z1'])
f['dset1'].dims[1].attach_scale(f['y1'])
f['dset1'].dims[2].attach_scale(f['x1'])
f['dset1'].dims[2].attach_scale(f['x2'])

# dataset with no dimension entries
f.create_dataset('dset2', data=np.ones((4, 3, 2)), **common_args)

f.close()
