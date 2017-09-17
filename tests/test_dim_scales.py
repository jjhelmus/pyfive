""" Unit tests for pyfive dimension scales. """

import os

from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DIM_SCALES_HDF5_FILE = os.path.join(DIRNAME, './dim_scales.hdf5')


def test_dim_labels():

    with pyfive.File(DIM_SCALES_HDF5_FILE) as hfile:

        # dataset with dimension labels
        dims = hfile['dset1'].dims
        assert dims[0].label == 'z'
        assert dims[1].label == 'y'
        assert dims[2].label == 'x'

        # dataset with no dimension labels
        dims = hfile['dset2'].dims
        assert dims[0].label == ''
        assert dims[1].label == ''
        assert dims[2].label == ''


def test_dim_scales():

    with pyfive.File(DIM_SCALES_HDF5_FILE) as hfile:

        # dataset with dimension scales
        dims = hfile['dset1'].dims

        assert len(dims) == 3

        assert len(dims[0]) == 1
        assert len(dims[1]) == 1
        assert len(dims[2]) == 2

        assert dims[0][0].name == '/z1'
        assert dims[1][0].name == '/y1'
        assert dims[2][0].name == '/x1'
        assert dims[2][1].name == '/x2'

        assert_array_equal(dims[0][0][:], [0, 10, 20, 30])
        assert_array_equal(dims[1][0][:], [3, 4, 5])
        assert_array_equal(dims[2][0][:], [1, 2])
        assert_array_equal(dims[2][1][:], [99, 98])

        # dataset with no dimension scales
        dims = hfile['dset2'].dims

        assert len(dims) == 3

        assert len(dims[0]) == 0
        assert len(dims[1]) == 0
        assert len(dims[2]) == 0
