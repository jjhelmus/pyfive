""" Unit tests for pyfive using the filelike objects  """

import io
import os
import h5py

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


def test_read_latest_fileobj():

    with io.open(LATEST_HDF5_FILE, 'rb') as f:
        with pyfive.File(f) as hfile:

            assert hfile.filename == LATEST_HDF5_FILE

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
            ref_attr6 = u'Test' + b'\xc2\xa7'.decode('utf-8')
            assert dset3.attrs['attr6'] == ref_attr6
            assert isinstance(dset3.attrs['attr6'], string_type)


def write_compressed_tobytes(file_like):
    """ Make an HDF file for testing """
    
    f = h5py.File(file_like, 'w', libver='earliest')

    # gzip compressed dataset
    f.create_dataset('dataset1', shape=(21, 16), chunks=(2, 2), dtype='<u2',
        compression='gzip', shuffle=False,
        data=np.arange(21*16).reshape(21, 16), track_times=False)
    f.close()


def test_iobytes():
    tfile = io.BytesIO()
    write_compressed_tobytes(tfile)
    with pyfive.File(tfile) as hfile:
        ds1 = hfile['dataset1']
        shape = ds1.shape
        assert shape == (21,16)

