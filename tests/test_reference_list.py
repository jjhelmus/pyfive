""" Unit tests for pyfive's ability to deal with reference lists """
import os
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
NETCDF4_CLASSIC_FILE = os.path.join(DIRNAME, 'netcdf4_classic.nc')


def test_read_netcdf4_dims():

    with pyfive.File(NETCDF4_CLASSIC_FILE) as hfile:

        with warnings.catch_warnings(record=True) as caught_warnings:
            dimensions_x = hfile['x'].dims 
            if caught_warnings:
                for warning in caught_warnings:
                    print('Caught warning ', warning)
                raise NotImplementedError('We need to fix this warning!')