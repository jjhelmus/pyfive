""" Test pyfive's abililty to read multidimensional datasets. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_COMPRESSED_HDF5_FILE = os.path.join(DIRNAME, 'compressed.hdf5')


def test_compressed_dataset():

    with pyfive.File(DATASET_COMPRESSED_HDF5_FILE) as hfile:

        # check data
        dset1 = hfile['dataset1']
        assert dset1.shape == (21, 16)
        assert dset1.dtype == np.dtype('u2')
        assert dset1.compression == 'gzip'
        assert dset1.compression_opts == 4
        assert dset1.shuffle is False
        assert_array_equal(dset1[:], np.arange(21*16).reshape((21, 16)))

        dset2 = hfile['dataset2']
        assert dset2.shape == (21, 16)
        assert dset2.dtype == np.dtype('i4')
        assert dset2.compression == 'gzip'
        assert dset2.compression_opts == 4
        assert dset2.shuffle is True
        assert_array_equal(dset2[:], np.arange(21*16).reshape((21, 16)))

        dset3 = hfile['dataset3']
        assert dset3.shape == (21, 16)
        assert dset3.dtype == np.dtype('f8')
        assert dset3.compression is None
        assert dset3.compression_opts is None
        assert dset3.shuffle is True
        assert_array_equal(dset2[:], np.arange(21*16).reshape((21, 16)))
