""" Unit tests for pyfive's ability to read a file with filter_pipeline v2
(as is found in some new netCDF4 files) """
import os
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
# test file from: https://github.com/Unidata/netcdf4-python/blob/master/examples/data/prmsl.2011.nc
NETCDF4_FILE = os.path.join(DIRNAME, 'prmsl.2011.nc')


def test_filter_pipeline_descr_v2():

    with pyfive.File(NETCDF4_FILE) as hfile:

        # group
        assert 'prmsl' in hfile
        d = hfile['prmsl']
        assert d.chunks == (1, 91, 180)
        assert d.shuffle == True
        assert d.compression == 'gzip'
        assert d.shape == (365, 91, 180)

