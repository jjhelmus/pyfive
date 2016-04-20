
import numpy as np
from numpy.testing import assert_array_equal

import ppyhdf5
import h5py

def test_read_basic_example():

    # reading with HDF5
    hfile = h5py.File('basic_example.hdf5', 'r')
    assert hfile['/example'].attrs['foo'] == 99.5
    assert hfile['/example'].attrs['bar'] == 42
    np.testing.assert_array_equal(
        hfile['/example'][:],
        np.arange(100, dtype='int32'))
    assert hfile['/example'].dtype == np.dtype('int32')
    assert hfile['/example'].shape == (100, )
    hfile.close()

    py_hfile = ppyhdf5.HDF5File('basic_example.hdf5')
    #assert py_hfile['/example'].attrs['foo'] == 99.5
    #assert py_hfile['/example'].attrs['bar'] == 42
    #np.testing.assert_array_equal(
    #    py_hfile['/example'][:],
    #    np.arange(100, dtype='int32'))
    #assert py_hfile['/example'].dtype == np.dtype('int32')
    #assert py_hfile['/example'].shape == (100, )
    py_hfile.close()


