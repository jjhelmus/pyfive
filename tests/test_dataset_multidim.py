""" Test pyfive's abililty to read multidimensional datasets. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_MULTIDIM_HDF5_FILE = os.path.join(DIRNAME, 'dataset_multidim.hdf5')


def test_multidim_datasets():

    hfile = pyfive.HDF5File(DATASET_MULTIDIM_HDF5_FILE)
    dsets = hfile.datasets

    # check shapes
    assert dsets['a'].data.shape == (2, )
    assert dsets['b'].data.shape == (2, 3)
    assert dsets['c'].data.shape == (2, 3, 4)
    assert dsets['d'].data.shape == (2, 3, 4, 5)

    # check data
    assert_array_equal(dsets['a'].data, np.arange(2).reshape((2, )))
    assert_array_equal(dsets['b'].data, np.arange(6).reshape((2, 3)))
    assert_array_equal(dsets['c'].data, np.arange(24).reshape((2, 3, 4)))
    assert_array_equal(dsets['d'].data, np.arange(120).reshape((2, 3, 4, 5)))

    hfile.close()
