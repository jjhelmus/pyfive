""" Test pyfive's abililty to read multidimensional datasets. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_MULTIDIM_HDF5_FILE = os.path.join(DIRNAME, 'dataset_multidim.hdf5')


def test_multidim_datasets():

    with pyfive.File(DATASET_MULTIDIM_HDF5_FILE) as hfile:

        # check shapes
        assert hfile['a'][:].shape == (2, )
        assert hfile['b'][:].shape == (2, 3)
        assert hfile['c'][:].shape == (2, 3, 4)
        assert hfile['d'][:].shape == (2, 3, 4, 5)

        # check data
        assert_array_equal(hfile['a'][:], np.arange(2).reshape((2, )))
        assert_array_equal(hfile['b'][:], np.arange(6).reshape((2, 3)))
        assert_array_equal(hfile['c'][:], np.arange(24).reshape((2, 3, 4)))
        assert_array_equal(hfile['d'][:], np.arange(120).reshape((2, 3, 4, 5)))
