""" Test pyfive's abililty to read datasets with a fletcher32 filter. """
import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import pyfive
from pyfive.btree import BTreeV1RawDataChunks

DIRNAME = os.path.dirname(__file__)
DATASET_FLETCHER_HDF5_FILE = os.path.join(DIRNAME, 'fletcher32.hdf5')


def test_fletcher32_datasets():

    with pyfive.File(DATASET_FLETCHER_HDF5_FILE) as hfile:

        # check data
        dset1 = hfile['dataset1']
        assert_array_equal(dset1[:], np.arange(4*4).reshape((4, 4)))
        assert dset1.chunks == (2, 2)

        # check data
        dset2 = hfile['dataset2']
        assert_array_equal(dset2[:], np.arange(3))
        assert dset2.chunks == (3, )

        # check attribute
        assert dset1.fletcher32


class TestChunkFletcher32(unittest.TestCase):

    def test_fletcher32_invalid(self):
        bad_chunk = b'\x00\x00\x00\x01'
        with self.assertRaises(ValueError) as context:
            BTreeV1RawDataChunks._verify_fletcher32(bad_chunk)
