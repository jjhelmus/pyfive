""" Unit tests for pyfive. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_DATATYPES_HDF5_FILE = os.path.join(DIRNAME, 'dataset_datatypes.hdf5')


def test_signed_int_dataset_datatypes():

    with pyfive.File(DATASET_DATATYPES_HDF5_FILE) as hfile:

        # check data
        ref_data = -np.arange(4)
        assert_array_equal(hfile['int08_little'][:], ref_data)
        assert_array_equal(hfile['int16_little'][:], ref_data)
        assert_array_equal(hfile['int32_little'][:], ref_data)
        assert_array_equal(hfile['int64_little'][:], ref_data)

        assert_array_equal(hfile['int08_big'][:], ref_data)
        assert_array_equal(hfile['int16_big'][:], ref_data)
        assert_array_equal(hfile['int32_big'][:], ref_data)
        assert_array_equal(hfile['int64_big'][:], ref_data)

        # check dtype
        assert hfile['int08_little'].dtype == np.dtype('<i1')
        assert hfile['int16_little'].dtype == np.dtype('<i2')
        assert hfile['int32_little'].dtype == np.dtype('<i4')
        assert hfile['int64_little'].dtype == np.dtype('<i8')

        assert hfile['int08_big'].dtype == np.dtype('>i1')
        assert hfile['int16_big'].dtype == np.dtype('>i2')
        assert hfile['int32_big'].dtype == np.dtype('>i4')
        assert hfile['int64_big'].dtype == np.dtype('>i8')


def test_unsigned_int_dataset_datatypes():

    with pyfive.File(DATASET_DATATYPES_HDF5_FILE) as hfile:

        # check data
        ref_data = np.arange(4)
        assert_array_equal(hfile['uint08_little'][:], ref_data)
        assert_array_equal(hfile['uint16_little'][:], ref_data)
        assert_array_equal(hfile['uint32_little'][:], ref_data)
        assert_array_equal(hfile['uint64_little'][:], ref_data)

        assert_array_equal(hfile['uint08_big'][:], ref_data)
        assert_array_equal(hfile['uint16_big'][:], ref_data)
        assert_array_equal(hfile['uint32_big'][:], ref_data)
        assert_array_equal(hfile['uint64_big'][:], ref_data)

        # check dtype
        assert hfile['uint08_little'].dtype == np.dtype('<u1')
        assert hfile['uint16_little'].dtype == np.dtype('<u2')
        assert hfile['uint32_little'].dtype == np.dtype('<u4')
        assert hfile['uint64_little'].dtype == np.dtype('<u8')

        assert hfile['uint08_big'].dtype == np.dtype('>u1')
        assert hfile['uint16_big'].dtype == np.dtype('>u2')
        assert hfile['uint32_big'].dtype == np.dtype('>u4')
        assert hfile['uint64_big'].dtype == np.dtype('>u8')


def test_float_dataset_datatypes():

    with pyfive.File(DATASET_DATATYPES_HDF5_FILE) as hfile:

        # check data
        ref_data = np.arange(4)
        assert_array_equal(hfile['float32_little'][:], ref_data)
        assert_array_equal(hfile['float64_little'][:], ref_data)

        assert_array_equal(hfile['float32_big'][:], ref_data)
        assert_array_equal(hfile['float64_big'][:], ref_data)

        # check dtype
        assert hfile['float32_little'].dtype == np.dtype('<f4')
        assert hfile['float64_little'].dtype == np.dtype('<f8')

        assert hfile['float32_big'].dtype == np.dtype('>f4')
        assert hfile['float64_big'].dtype == np.dtype('>f8')
