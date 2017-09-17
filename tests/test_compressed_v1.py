""" Test pyfive's abililty to read multidimensional datasets, version 1. """
import os

import numpy as np

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_COMPRESSED_HDF5_FILE = os.path.join(DIRNAME, 'compressed_v1.hdf5')


def test_compressed_v1_dataset():

    with pyfive.File(DATASET_COMPRESSED_HDF5_FILE) as hfile:

        # check data
        dset1 = hfile['temperature']
        assert dset1.shape == (816852,)
        assert dset1.dtype == np.dtype('>f4')
        assert dset1.compression == 'gzip'
        assert dset1.compression_opts == 4
        assert dset1.shuffle is False
        assert dset1[0] == 73.15625
        assert dset1[-1] == 85.71875
