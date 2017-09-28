""" Test pyfive's Dataset fillvalue attribute. """
import os

import pyfive

DIRNAME = os.path.dirname(__file__)
DATASET_FILLVALUE_EARLIEST_HDF5_FILE = os.path.join(
    DIRNAME, 'fillvalue_earliest.hdf5')
DATASET_FILLVALUE_LATEST_HDF5_FILE = os.path.join(
    DIRNAME, 'fillvalue_latest.hdf5')


def test_dataset_fillvalue_earliest():
    with pyfive.File(DATASET_FILLVALUE_EARLIEST_HDF5_FILE) as hfile:
        assert hfile['dset1'].fillvalue == 42
        assert hfile['dset2'].fillvalue == 0
        assert abs(hfile['dset3'].fillvalue - 99.5) < 0.05


def test_dataset_fillvalue_latest():
    with pyfive.File(DATASET_FILLVALUE_LATEST_HDF5_FILE) as hfile:
        assert hfile['dset1'].fillvalue == 42
        assert hfile['dset2'].fillvalue == 0
        assert abs(hfile['dset3'].fillvalue - 99.5) < 0.05
