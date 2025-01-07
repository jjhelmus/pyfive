""" Unit tests for pyfive's ability to read a NetCDF4 Classic file with an unlimited dimension"""
import os
import warnings

import numpy as np
from numpy.testing import assert_array_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
NETCDF4_UNLIMITED_FILE = os.path.join(DIRNAME, 'netcdf4_unlimited.nc')
NETCDF4_EMPTY_UNLIMITED_FILE = os.path.join(DIRNAME, 'netcdf4_empty_unlimited.nc')
H5NETCDF_FILE = os.path.join(DIRNAME, 'h5netcdf_test.hdf5')

def test_read_netcdf4_unlimited():
    """" This works"""

    with pyfive.File(NETCDF4_UNLIMITED_FILE) as hfile:

        # dataset
        var1 = hfile['foo_unlimited']
        assert var1.dtype == np.dtype('<f8')
        assert_array_equal(var1[:], np.ones((4,1)))
       
def test_read_netcdf4_empty_unlimited():
    "This does not work currently. Why not?"
    # This is one example of the sort of problem we see with the H6NetCDF file.
    with pyfive.File(NETCDF4_EMPTY_UNLIMITED_FILE) as hfile:

        # dataset
        var1 = hfile['foo_unlimited']
        assert var1.dtype == np.dtype('<f8')
        print (var1[:])

def test_h5netcdf_file():
    """ This doesn't work either. Why not? """

    with pyfive.File(H5NETCDF_FILE) as hfile:

        # dataset
        var1 = hfile['empty']
        print(var1.shape)
        print(var1[:])