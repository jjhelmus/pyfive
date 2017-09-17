""" Unit tests for pyfive using the earliest library version """
import os

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import pyfive

DIRNAME = os.path.dirname(__file__)
LATEST_HDF5_FILE = os.path.join(DIRNAME, 'latest.hdf5')

# Polygot string type for representing unicode
try:
    string_type = unicode
except NameError:
    string_type = str


def test_read_latest():

    with pyfive.File(LATEST_HDF5_FILE) as hfile:

        # root
        assert hfile.attrs['attr1'] == -123
        assert hfile.attrs['attr1'].dtype == np.dtype('int32')

        dset1 = hfile['dataset1']
        assert_array_equal(dset1[:], np.arange(4))
        assert dset1.dtype == np.dtype('<i4')
        assert dset1.attrs['attr2'] == 130
        assert dset1.attrs['attr2'].dtype == np.dtype('uint8')

        # group
        grp = hfile['/group1']
        assert_almost_equal(grp.attrs['attr3'], 12.34, 2)
        assert grp.attrs['attr3'].dtype == np.dtype('float32')

        dset2 = grp['dataset2']
        assert_array_equal(dset2[:], np.arange(4))
        assert dset2.dtype == np.dtype('>u8')
        assert dset2.attrs['attr4'] == b'Hi'
        assert dset2.attrs['attr4'].dtype == np.dtype('|S2')

        # sub-group
        subgroup = grp['subgroup1']
        assert subgroup.attrs['attr5'] == b'Test'
        assert isinstance(subgroup.attrs['attr5'], bytes)

        dset3 = subgroup['dataset3']
        assert_array_equal(dset2[:], np.arange(4))
        assert dset3.dtype == np.dtype('<f4')
        assert dset3.attrs['attr6'] == u'Test' + b'\xc2\xa7'.decode('utf-8')
        assert isinstance(dset3.attrs['attr6'], string_type)
