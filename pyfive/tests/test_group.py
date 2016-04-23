import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive


def test_file_with_group():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'group_example.hdf5')
    hfile = pyfive.HDF5File(filename)

    # root group attribute
    assert 'alice' in hfile.attrs
    assert hfile.attrs['alice'] == 12

    # example dataset
    assert 'example' in hfile.datasets
    assert 'example' not in hfile.groups
    dset = hfile.datasets['example']

    assert 'bar' in dset.attrs
    assert 'foo' in dset.attrs
    assert dset.attrs['bar'] == 42
    assert dset.attrs['foo'] == 99.5

    data = dset.get_data()
    assert data.dtype == np.dtype('int32')
    assert data.shape == (100, )
    assert_array_equal(data, np.arange(100, dtype='int32'))

    # subgroup one
    assert 'subgroup_one' not in hfile.datasets
    assert 'subgroup_one' in hfile.groups
    grp = hfile.groups['subgroup_one']
    assert 'bob' in grp.attrs
    assert grp.attrs['bob'] == 13

    # subgroup_one/grp_one_array
    dset = grp.datasets['grp_one_array']
    assert 'carol' in dset.attrs
    assert dset.attrs['carol'] == 14

    data = dset.get_data()
    assert data.dtype == np.dtype('float32')
    assert data.shape == (10, )
    assert_array_equal(data, np.arange(10) + 55)

    hfile.close()
