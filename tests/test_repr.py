""" Unit tests for pyfive's class __repr__ methods.n """

import os

import pyfive

DIRNAME = os.path.dirname(__file__)
EARLIEST_HDF5_FILE = os.path.join(DIRNAME, 'earliest.hdf5')


def test_repr():

    with pyfive.File(EARLIEST_HDF5_FILE) as hfile:
        hfile_str = '<HDF5 file "earliest.hdf5" (mode r)>'
        assert str(hfile) == hfile_str

        group1 = hfile['group1']
        group1_str = '<HDF5 group "/group1" (2 members)>'
        assert str(group1) == group1_str

        dataset1 = hfile['dataset1']
        dataset1_str = '<HDF5 dataset "dataset1": shape (4,), type "<i4">'
        assert str(dataset1) == dataset1_str

        subgroup1 = group1['subgroup1']
        subgroup1_str = '<HDF5 group "/group1/subgroup1" (1 members)>'
        assert str(subgroup1) == subgroup1_str

        dataset2 = group1['dataset2']
        dataset2_str = '<HDF5 dataset "dataset2": shape (4,), type ">u8">'
        assert str(dataset2) == dataset2_str
