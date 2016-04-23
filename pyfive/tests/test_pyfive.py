""" Unit tests for pyfive. """
import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive
import h5py

DIRNAME = os.path.dirname(__file__)
BASIC_HDF5_FILE = os.path.join(DIRNAME, 'basic_example.hdf5')
BASIC_NETCDF4_FILE = os.path.join(DIRNAME, 'basic_example.nc')


def test_read_basic_example():

    # reading with HDF5
    hfile = h5py.File(BASIC_HDF5_FILE, 'r')
    assert hfile['/example'].attrs['foo'] == 99.5
    assert hfile['/example'].attrs['bar'] == 42
    np.testing.assert_array_equal(
        hfile['/example'][:],
        np.arange(100, dtype='int32'))
    assert hfile['/example'].dtype == np.dtype('int32')
    assert hfile['/example'].shape == (100, )
    hfile.close()

    # reading with pyfive
    hfile = pyfive.HDF5File(BASIC_HDF5_FILE)
    assert 'example' in hfile.datasets

    dset = hfile.datasets['example']

    assert 'bar' in dset.attrs
    assert 'foo' in dset.attrs
    assert dset.attrs['bar'] == 42
    assert dset.attrs['foo'] == 99.5

    data = dset.get_data()
    assert data.dtype == np.dtype('int32')
    assert data.shape == (100, )
    assert_array_equal(data, np.arange(100, dtype='int32'))

    hfile.close()
