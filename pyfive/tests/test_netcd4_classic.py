""" Unit tests for pyfive's ability to read a NetCDF4 Classic file """
import os

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
NETCDF4_CLASSIC_FILE = os.path.join(DIRNAME, 'netcdf4_classic.nc')


def test_read_netcdf4_classic():

    hfile = pyfive.HDF5File(NETCDF4_CLASSIC_FILE)

    # attributes
    assert hfile.attrs['attr1'] == -123
    assert hfile.attrs['attr2'] == 130

    # dataset
    var1 = hfile.datasets['var1']
    assert var1.data.dtype == np.dtype('<i4')
    assert_array_equal(var1.data, np.arange(4))

    # dataset attributes
    assert_almost_equal(var1.attrs['attr3'], 12.34, 2)
    assert var1.attrs['attr4'] == b'Hi'

    hfile.close()
