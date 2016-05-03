#! /usr/bin/env python
""" Create a HDF5 file with layers of groups. """
import h5py
import numpy as np


f = h5py.File('groups.hdf5', 'w')

# groups
grp1 = f.create_group('group1')
grp2 = f.create_group('group2')

# sub-groups
sub1 = grp2.create_group('subgroup1')
sub2 = grp2.create_group('subgroup2')

# sub-sub-groups
sub2.create_group('sub_subgroup1')
sub2.create_group('sub_subgroup2')
sub2.create_group('sub_subgroup3')

f.close()
