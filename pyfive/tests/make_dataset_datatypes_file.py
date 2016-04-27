#! /usr/bin/env python
""" Create a HDF5 file with datasets of many datatypes . """
import h5py
import numpy as np

f = h5py.File('dataset_datatypes.hdf5', 'w')

# signed intergers
common_signed_args = {
    'shape': (4, ),
    'data': -np.arange(4),
    'track_times': False,
}

f.create_dataset('int08_little', dtype='<i1', **common_signed_args)
f.create_dataset('int16_little', dtype='<i2', **common_signed_args)
f.create_dataset('int32_little', dtype='<i4', **common_signed_args)
f.create_dataset('int64_little', dtype='<i8', **common_signed_args)

f.create_dataset('int08_big', dtype='>i1', **common_signed_args)
f.create_dataset('int16_big', dtype='>i2', **common_signed_args)
f.create_dataset('int32_big', dtype='>i4', **common_signed_args)
f.create_dataset('int64_big', dtype='>i8', **common_signed_args)

# unsigned intergers
common_unsigned_args = {
    'shape': (4, ),
    'data': np.arange(4),
    'track_times': False,
}

f.create_dataset('uint08_little', dtype='<u1', **common_unsigned_args)
f.create_dataset('uint16_little', dtype='<u2', **common_unsigned_args)
f.create_dataset('uint32_little', dtype='<u4', **common_unsigned_args)
f.create_dataset('uint64_little', dtype='<u8', **common_unsigned_args)

f.create_dataset('uint08_big', dtype='>u1', **common_unsigned_args)
f.create_dataset('uint16_big', dtype='>u2', **common_unsigned_args)
f.create_dataset('uint32_big', dtype='>u4', **common_unsigned_args)
f.create_dataset('uint64_big', dtype='>u8', **common_unsigned_args)

# floating point
common_float_args = {
    'shape': (4, ),
    'data': np.arange(4),
    'track_times': False,
}

f.create_dataset('float32_little', dtype='<f4', **common_float_args)
f.create_dataset('float64_little', dtype='<f8', **common_float_args)

f.create_dataset('float32_big', dtype='>f4', **common_float_args)
f.create_dataset('float64_big', dtype='>f8', **common_float_args)

f.close()
