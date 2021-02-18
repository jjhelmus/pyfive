#! /usr/bin/env python
""" Create a HDF5 file with new-style groups. """
import h5py
import numpy as np


f = h5py.File('new_style_groups.hdf5', 'w', track_order=True)
for i in range(9):
    f.create_group('group' + str(i))
