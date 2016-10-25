#! /usr/bin/env python
""" Create a HDF5 file with references. """
import h5py
import numpy as np

f = h5py.File('references.hdf5', 'w')

# some HDF5 objects for testing
f.attrs.create('root_attr', 123)

dset1 = f.create_dataset(
    'dataset1', shape=(4, ), dtype='<i4', data=np.arange(4), track_times=False)
dset1.attrs.create('dset_attr', 456)
region_ref = dset1.regionref[::2]

grp = f.create_group('group1')
grp.attrs.create('group_attr', 789)

# references
f.attrs['root_group_reference'] = f.ref
f.attrs['dataset1_reference'] = dset1.ref
f.attrs['group1_reference'] = grp.ref
f.attrs['dataset1_region_reference'] = region_ref

# array of references
ref_dtype = h5py.special_dtype(ref=h5py.Reference)

ref_dataset = f.create_dataset(
    "ref_dataset", (4,), dtype=ref_dtype, track_times=False)
ref_dataset[0] = f.ref
ref_dataset[1] = dset1.ref
ref_dataset[2] = grp.ref
# ref_dataset[3] is a Null reference

chunked_ref_dataset = f.create_dataset(
    "chunked_ref_dataset", (4,), chunks=(2, ), dtype=ref_dtype,
    track_times=False)
chunked_ref_dataset[0] = f.ref
chunked_ref_dataset[1] = dset1.ref
chunked_ref_dataset[2] = grp.ref
# chunked_ref_dataset[3] is a Null reference

regionref_dtype = h5py.special_dtype(ref=h5py.RegionReference)

regionref_dataset = f.create_dataset(
    "regionref_dataset", (2,), dtype=regionref_dtype, track_times=False)
regionref_dataset[0] = region_ref

chunked_regionref_dataset = f.create_dataset(
    "chunked_regionref_dataset", (2,), chunks=(1, ), dtype=regionref_dtype,
    track_times=False)
chunked_regionref_dataset[0] = region_ref
# chunked_regionref_dataset[1] is a Null reference

f.close()
