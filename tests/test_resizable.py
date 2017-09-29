""" Test pyfive's abililty to read resizable datasets. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_RESIZABLE_HDF5_FILE = os.path.join(DIRNAME, 'resizable.hdf5')


def test_resizable_dataset():

    with pyfive.File(DATASET_RESIZABLE_HDF5_FILE) as hfile:

        dset1 = hfile['dataset1']
        assert_array_equal(dset1[:], np.arange(4 * 6).reshape((4, 6)))
        assert dset1.dtype == '<f8'

        dset2 = hfile['dataset2']
        assert_array_equal(dset2[:], np.arange(10 * 5).reshape((10, 5)))
        assert dset2.dtype == '<i4'

        dset3 = hfile['dataset3']
        assert_array_equal(dset3[:], np.arange(8 * 4).reshape((8, 4)))
        assert dset3.dtype == '>i2'
