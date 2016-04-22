import os

import numpy as np
from numpy.testing import assert_array_equal

import pyfive


def test_file_with_group():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'group_example.hdf5')
    hfile = pyfive.HDF5File(filename)

    # root group attribute
    attrs = hfile.get_attributes()
    assert 'alice' in attrs
    assert attrs['alice'] == 12

    # example dataset
    assert 'example' in hfile.datasets
    assert 'example' not in hfile.groups
    dset = hfile.datasets['example']

    attrs = dset.get_attributes()
    assert 'bar' in attrs
    assert 'foo' in attrs
    assert attrs['bar'] == 42
    assert attrs['foo'] == 99.5

    data = dset.get_data()
    assert data.dtype == np.dtype('int32')
    assert data.shape == (100, )
    assert_array_equal(data, np.arange(100, dtype='int32'))

    # subgroup one
    assert 'subgroup_one' not in hfile.datasets
    assert 'subgroup_one' in hfile.groups
    grp = hfile.groups['subgroup_one']
    attrs = grp.get_attributes()
    assert 'bob' in attrs
    assert attrs['bob'] == 13

    # subgroup_one/grp_one_array
    dset = grp.datasets['grp_one_array']
    attrs = dset.get_attributes()
    assert 'carol' in attrs
    assert attrs['carol'] == 14

    data = dset.get_data()
    assert data.dtype == np.dtype('float32')
    assert data.shape == (10, )
    assert_array_equal(data, np.arange(10) + 55)

    hfile.close()
