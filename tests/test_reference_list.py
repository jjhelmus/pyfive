""" Unit tests for pyfive's ability to deal with reference lists """
import os
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import pyfive
import h5py

DIRNAME = os.path.dirname(__file__)
NETCDF4_CLASSIC_FILE = os.path.join(DIRNAME, 'netcdf4_classic.nc')


def test_read_netcdf4_dims():

    # We want to know about this warning and deal with it
    #warnings.simplefilter('error', UserWarning)

    # not using a context manager so we can compare and contrast in debugging
    hfile1 = h5py.File(NETCDF4_CLASSIC_FILE)
    dimensions_x1 = hfile1['x'].dims
    dimensions_v1 = hfile1['var1'].dims

    hfile2 = pyfive.File(NETCDF4_CLASSIC_FILE)
    dimensions_x2 = hfile2['x'].dims
    dimensions_v2 = hfile2['var1'].dims

    # as we created no explicit data for this dimension, this is the case where
    # getitme goes to storage and finds an UNDEFINED_ADDRESS and returns zeros.
    y = hfile2['x'][:]
    
    assert len(dimensions_v1) == len(dimensions_v2)
    assert len(dimensions_x1) == len(dimensions_x2)

    # The dimension scale spec is here: https://support.hdfgroup.org/documentation/hdf5-docs/hdf5_topics/H5DS_Spec.pdf
    # The issue is that we don't support reference_lists.
    # But we don't know if this matters or not, given this is failing on the dimensions of a dimension
    # ChatGPT says: 
    #  
    # - NetCDF4 Classic files often abstract away dimensions, so their behavior in HDF5 tools 
    #   like h5py may not always align with expectations for standard HDF5 datasets.
