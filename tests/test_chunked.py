""" Test pyfive's abililty to read multidimensional datasets. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_CHUNKED_HDF5_FILE = os.path.join(DIRNAME, 'chunked.hdf5')


def test_chunked_dataset():

    with pyfive.File(DATASET_CHUNKED_HDF5_FILE) as hfile:

        # check data
        dset1 = hfile['dataset1']
        assert_array_equal(dset1[:], np.arange(21*16).reshape((21, 16)))
        assert dset1.chunks == (2, 2)
