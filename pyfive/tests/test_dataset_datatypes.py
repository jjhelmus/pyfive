""" Unit tests for pyfive. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_DATATYPES_HDF5_FILE = os.path.join(DIRNAME, 'dataset_datatypes.hdf5')


def test_signed_int_dataset_datatypes():

    hfile = pyfive.HDF5File(DATASET_DATATYPES_HDF5_FILE)

    # check data
    ref_data = -np.arange(4)
    assert_array_equal(hfile.datasets['int08_little'].data, ref_data)
    assert_array_equal(hfile.datasets['int16_little'].data, ref_data)
    assert_array_equal(hfile.datasets['int32_little'].data, ref_data)
    assert_array_equal(hfile.datasets['int64_little'].data, ref_data)

    assert_array_equal(hfile.datasets['int08_big'].data, ref_data)
    assert_array_equal(hfile.datasets['int16_big'].data, ref_data)
    assert_array_equal(hfile.datasets['int32_big'].data, ref_data)
    assert_array_equal(hfile.datasets['int64_big'].data, ref_data)

    # check dtype
    assert hfile.datasets['int08_little'].data.dtype == np.dtype('<i1')
    assert hfile.datasets['int16_little'].data.dtype == np.dtype('<i2')
    assert hfile.datasets['int32_little'].data.dtype == np.dtype('<i4')
    assert hfile.datasets['int64_little'].data.dtype == np.dtype('<i8')

    assert hfile.datasets['int08_big'].data.dtype == np.dtype('>i1')
    assert hfile.datasets['int16_big'].data.dtype == np.dtype('>i2')
    assert hfile.datasets['int32_big'].data.dtype == np.dtype('>i4')
    assert hfile.datasets['int64_big'].data.dtype == np.dtype('>i8')

    hfile.close()


def test_unsigned_int_dataset_datatypes():

    hfile = pyfive.HDF5File(DATASET_DATATYPES_HDF5_FILE)

    # check data
    ref_data = np.arange(4)
    assert_array_equal(hfile.datasets['uint08_little'].data, ref_data)
    assert_array_equal(hfile.datasets['uint16_little'].data, ref_data)
    assert_array_equal(hfile.datasets['uint32_little'].data, ref_data)
    assert_array_equal(hfile.datasets['uint64_little'].data, ref_data)

    assert_array_equal(hfile.datasets['uint08_big'].data, ref_data)
    assert_array_equal(hfile.datasets['uint16_big'].data, ref_data)
    assert_array_equal(hfile.datasets['uint32_big'].data, ref_data)
    assert_array_equal(hfile.datasets['uint64_big'].data, ref_data)

    # check dtype
    assert hfile.datasets['uint08_little'].data.dtype == np.dtype('<u1')
    assert hfile.datasets['uint16_little'].data.dtype == np.dtype('<u2')
    assert hfile.datasets['uint32_little'].data.dtype == np.dtype('<u4')
    assert hfile.datasets['uint64_little'].data.dtype == np.dtype('<u8')

    assert hfile.datasets['uint08_big'].data.dtype == np.dtype('>u1')
    assert hfile.datasets['uint16_big'].data.dtype == np.dtype('>u2')
    assert hfile.datasets['uint32_big'].data.dtype == np.dtype('>u4')
    assert hfile.datasets['uint64_big'].data.dtype == np.dtype('>u8')

    hfile.close()


def test_float_dataset_datatypes():

    hfile = pyfive.HDF5File(DATASET_DATATYPES_HDF5_FILE)

    # check data
    ref_data = np.arange(4)
    assert_array_equal(hfile.datasets['float32_little'].data, ref_data)
    assert_array_equal(hfile.datasets['float64_little'].data, ref_data)

    assert_array_equal(hfile.datasets['float32_big'].data, ref_data)
    assert_array_equal(hfile.datasets['float64_big'].data, ref_data)

    # check dtype
    assert hfile.datasets['float32_little'].data.dtype == np.dtype('<f4')
    assert hfile.datasets['float64_little'].data.dtype == np.dtype('<f8')

    assert hfile.datasets['float32_big'].data.dtype == np.dtype('>f4')
    assert hfile.datasets['float64_big'].data.dtype == np.dtype('>f8')

    hfile.close()
