""" Unit tests for pyfive. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
ATTR_DATATYPES_HDF5_FILE = os.path.join(DIRNAME, 'attr_datatypes.hdf5')


def test_numeric_scalar_attr_datatypes():

    with pyfive.File(ATTR_DATATYPES_HDF5_FILE) as hfile:

        assert hfile.attrs['int08_little'] == -123
        assert hfile.attrs['int16_little'] == -123
        assert hfile.attrs['int32_little'] == -123
        assert hfile.attrs['int64_little'] == -123

        # These are 2**(size_in_bytes-1)+2 which could not be stored in
        # signed type of the same size
        assert hfile.attrs['uint08_little'] == 130
        assert hfile.attrs['uint16_little'] == 32770
        assert hfile.attrs['uint32_little'] == 2147483650
        assert hfile.attrs['uint64_little'] == 9223372036854775810

        assert hfile.attrs['int08_big'] == -123
        assert hfile.attrs['int16_big'] == -123
        assert hfile.attrs['int32_big'] == -123
        assert hfile.attrs['int64_big'] == -123

        assert hfile.attrs['uint08_big'] == 130
        assert hfile.attrs['uint16_big'] == 32770
        assert hfile.attrs['uint32_big'] == 2147483650
        assert hfile.attrs['uint64_big'] == 9223372036854775810

        assert hfile.attrs['float32_little'] == 123.
        assert hfile.attrs['float64_little'] == 123.

        assert hfile.attrs['float32_big'] == 123.
        assert hfile.attrs['float64_big'] == 123.


def test_complex_scalar_attr_datatypes():

    with pyfive.File(ATTR_DATATYPES_HDF5_FILE) as hfile:

        assert hfile.attrs['complex64_little'] == (123 + 456j)
        assert hfile.attrs['complex128_little'] == (123 + 456j)

        assert hfile.attrs['complex64_big'] == (123 + 456j)
        assert hfile.attrs['complex128_big'] == (123 + 456j)


def test_string_scalar_attr_datatypes():

    with pyfive.File(ATTR_DATATYPES_HDF5_FILE) as hfile:

        assert hfile.attrs['string_one'] == b'H'
        assert hfile.attrs['string_two'] == b'Hi'

        assert hfile.attrs['vlen_string'] == b'Hello'
        assert hfile.attrs['vlen_unicode'] == (
            u'Hello' + b'\xc2\xa7'.decode('utf-8'))


def test_numeric_array_attr_datatypes():

    with pyfive.File(ATTR_DATATYPES_HDF5_FILE) as hfile:

        assert_array_equal(hfile.attrs['int32_array'], [-123, 45])
        assert_array_equal(hfile.attrs['uint64_array'], [12, 34])
        assert_array_equal(hfile.attrs['float32_array'], [123, 456])

        assert hfile.attrs['int32_array'].dtype == np.dtype('<i4')
        assert hfile.attrs['uint64_array'].dtype == np.dtype('>u8')
        assert hfile.attrs['float32_array'].dtype == np.dtype('<f4')

        assert hfile.attrs['vlen_str_array'][0] == b'Hello'
        assert hfile.attrs['vlen_str_array'][1] == b'World!'

        assert hfile.attrs['vlen_str_array'].dtype == np.dtype('S6')


def test_vlen_sequence_attr_datatypes():

    with pyfive.File(ATTR_DATATYPES_HDF5_FILE) as hfile:

        vlen_attr = hfile.attrs['vlen_int32']
        assert len(vlen_attr) == 2
        assert_array_equal(vlen_attr[0], [-1, 2])
        assert_array_equal(vlen_attr[1], [3, 4, 5])

        vlen_attr = hfile.attrs['vlen_uint64']
        assert len(vlen_attr) == 3
        assert_array_equal(vlen_attr[0], [1, 2])
        assert_array_equal(vlen_attr[1], [3, 4, 5])
        assert_array_equal(vlen_attr[2], [42])

        vlen_attr = hfile.attrs['vlen_float32']
        assert len(vlen_attr) == 3
        assert_array_equal(vlen_attr[0], [0])
        assert_array_equal(vlen_attr[1], [1, 2, 3])
        assert_array_equal(vlen_attr[2], [4, 5])
