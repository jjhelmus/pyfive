""" Unit tests for pyfive. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
ATTR_DATATYPES_HDF5_FILE = os.path.join(DIRNAME, 'attr_datatypes.hdf5')


def test_numeric_scalar_attr_datatypes():

    hfile = pyfive.HDF5File(ATTR_DATATYPES_HDF5_FILE)

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

    hfile.close()


def test_string_scalar_attr_datatypes():

    hfile = pyfive.HDF5File(ATTR_DATATYPES_HDF5_FILE)

    assert hfile.attrs['string_one'] == b'H'
    assert hfile.attrs['string_two'] == b'Hi'

    assert hfile.attrs['vlen_string'] == b'Hello'
    assert hfile.attrs['vlen_unicode'] == u'Hello' + chr(0x00A7)

    hfile.close()
