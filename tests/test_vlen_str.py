import h5py
import pyfive
import netCDF4 as nc
import io
import numpy as np
import os

def make_file_hdf5(our_file, vlen_strings):
   
    with h5py.File(our_file,'w') as hfile:
        
        dt = h5py.special_dtype(vlen=str)
        v = hfile.create_dataset("var_len_str", (2,), dtype=dt)
        v[:] = vlen_strings


def make_file_nc(file_like,m_array, inmemory=False):
  
    if inmemory:
        n = nc.Dataset(file_like, 'w', diskless=True)
    else:
        n = nc.Dataset(file_like, "w", format="NETCDF4")
    n.createDimension("time", 4)
    months = n.createVariable("months", str, ("time",))
    months[:] =  np.array(m_array, dtype="S8")
    if not inmemory:
        n.close()

def test_vlen_string_hdf5(tmp_path):

    #tfile = io.BytesIO()
    our_file = tmp_path/'h5py_vlen.hdf5'
    our_view = tmp_path/'h5py_vlen.txt'
    vlen_strings = ["foo","foobar"]
    make_file_hdf5(our_file, vlen_strings)
    os.system(f'h5dump {our_file} > {our_view}')
    with open(our_view,'r') as f:
        for line in f.readlines():
            print(line)

    with pyfive.File(our_file) as hfile:
         
        ds1 = hfile['var_len_str'][:]
        print(ds1)
        assert np.array_equal(ds1,vlen_strings)
    
def NOtest_vlen_string_nc1():
    """ this verson currently fails because netcdf4 is doing something odd in memory """

    t1file = io.BytesIO()
    m_array = ["January", "February", "March", "April"]
    make_file_nc(t1file,m_array, inmemory=True)

    with nc.Dataset(t1file,'r') as ncfile:
        ds1 = ncfile['months']
        assert np.array_equal(m_array, ds1) 

    with h5py.File(t1file) as pfile:
        ds1 = pfile['months']
        assert np.array_equal(m_array, ds1) 

    with pyfive.File(t1file) as hfile:
        ds1 = hfile['months']
        assert np.array_equal(m_array, ds1) 

def test_vlen_string_nc2(tmp_path):
    tfile = tmp_path / 'test_vlen_string.nc'
    m_array = ["January", "February", "March", "April"]
    make_file_nc(tfile, m_array)
    
    with nc.Dataset(tfile, 'r') as ncfile:
        ds1 = ncfile.variables['months'][:]
        assert np.array_equal(m_array, ds1.astype(str))
    
    with h5py.File(tfile, 'r') as pfile:
        ds1 = pfile['months'][:]
        assert np.array_equal(m_array, ds1.astype(str))
    
    with pyfive.File(tfile) as hfile:
        ds1 = hfile['months'][:]
        assert np.array_equal(m_array, ds1.astype(str))