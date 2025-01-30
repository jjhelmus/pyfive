""" Test pyfive's Dataset fillvalue attribute. """
import os

import netCDF4

import numpy as np

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


def make_uninitialized_data_file_nc(file_like):
    with netCDF4.Dataset(file_like, "w", format="NETCDF4") as n:
        n.createDimension("y", 3)
        n.createDimension("x", 4)

        # Default fillvalue
        n.createVariable("string", str, ("y", "x",))
        n.createVariable("char", 'S1', ("y", "x",))
        n.createVariable("int32", 'i4', ("y", "x",))
        n.createVariable("float64", 'f8', ("y", "x",))

        # Set fillvalue
        n.createVariable("string_2", str, ("y", "x",), fill_value='NA')
        n.createVariable("char_2", 'S1', ("y", "x",), fill_value='x')
        n.createVariable("int32_2", 'i4', ("y", "x",), fill_value=999)
        n.createVariable("float64_2", 'f8', ("y", "x",), fill_value=999.9)


def test_uninitialized_data(tmp_path):
    tfile = tmp_path / 'test_uninitialized_data.nc'
    make_uninitialized_data_file_nc(tfile)

    with pyfive.File(tfile, 'r') as pfile:
        assert np.array_equal(pfile['string'][1], [''] * 4)
        assert np.array_equal(pfile['char'][1], [b''] * 4)
        assert np.array_equal(pfile['int32'][1], [-2147483647] * 4)
        assert np.array_equal(pfile['float64'][1], [9.969209968386869e+36] * 4)

        assert np.array_equal(pfile['string_2'][1], ['NA'] * 4)
        assert np.array_equal(pfile['char_2'][1], [b'x'] * 4)
        assert np.array_equal(pfile['int32_2'][1], [999] * 4)
        assert np.array_equal(pfile['float64_2'][1], [999.9] * 4)
