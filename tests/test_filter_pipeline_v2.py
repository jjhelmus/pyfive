""" Unit tests for pyfive's ability to read a file with filter_pipeline v2
(as is found in some new netCDF4 files) """
import os

from numpy.testing import assert_almost_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
FILTER_PIPELINE_V2_FILE = os.path.join(DIRNAME, 'filter_pipeline_v2.hdf5')


def test_filter_pipeline_descr_v2():

    with pyfive.File(FILTER_PIPELINE_V2_FILE) as hfile:
        assert 'data' in hfile
        d = hfile['data']
        assert d.shape == (10,10,10)
        assert_almost_equal(d[0,0,0], 1.0)
