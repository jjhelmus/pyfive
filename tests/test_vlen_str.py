import h5py
import pyfive
import netCDF4 as nc
import io
import numpy as np
import os
import warnings

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

def make_contiguous_and_chunked_nc(our_file):
    m = ["January", "February", "March", "April", "May",
         "June", "July", "August", "September", "October",
         "November", "December"]

    with nc.Dataset(our_file, "w", format="NETCDF4") as n:
        n.createDimension("y", 3)
        n.createDimension("x", 4)

        # Contiguous variable
        months = n.createVariable("months", str, ("y", "x",))
        months.long_name = "string: Four months (contiguous)"
        months[...] = np.array(m, dtype="S9").reshape(3, 4)

        # Chunked variable
        months_chunked = n.createVariable("months_chunked", str, ("y", "x",),
                                   chunksizes=(2, 2))
        months_chunked.long_name = "string: Four months (chunked)"
        months_chunked[...] = np.array(m, dtype="U9").reshape(3, 4)


def make_pathological_nc(our_file):
    n = nc.Dataset(our_file, "w", format="NETCDF4")

    n.createDimension("dim1", 1)
    n.createDimension("time", 4)
    n.createDimension("lat", 2)
    n.createDimension("lon", 3)
    n.createDimension("strlen8", 8)
    n.createDimension("strlen7", 7)
    n.createDimension("strlen5", 5)
    n.createDimension("strlen3", 3)

    months = np.array(["January", "February", "March", "April"], dtype="S8")
   
    months_m = np.ma.array(
        months, dtype="S7", mask=[0, 1, 0, 0], fill_value=b""
    )

    numbers = np.array(
        [["one", "two", "three"], ["four", "five", "six"]], dtype="S5"
    )

    s_months4 = n.createVariable("s_months4", str, ("time",))
    s_months4.long_name = "string: Four months"
    s_months4[:] = months
    validation={'s_months4':s_months4[:]}

    s_months1 = n.createVariable("s_months1", str, ("dim1",))
    s_months1.long_name = "string: One month"
    s_months1[:] = np.array(["December"], dtype="S8")
    validation['s_months1'] = s_months1[:]

    s_months0 = n.createVariable("s_months0", str, ())
    s_months0.long_name = "string: One month (scalar)"
    s_months0[:] = np.array(["May"], dtype="S3")
    validation['s_months0'] = s_months0[:]

    s_numbers = n.createVariable("s_numbers", str, ("lat", "lon"))
    s_numbers.long_name = "string: Two dimensional"
    s_numbers[...] = numbers
    validation['s_numbers'] = s_numbers[:]

    s_months4m = n.createVariable("s_months4m", str, ("time",))
    s_months4m.long_name = "string: Four months (masked)"
    array = months.copy()
    array[1] = ""
    s_months4m[...] = array
    validation['s_months4m'] = s_months4m[...]

    c_months4 = n.createVariable("c_months4", "S1", ("time", "strlen8"))
    c_months4.long_name = "char: Four months"
    c_months4[:, :] = nc.stringtochar(months)
    validation['c_months4'] = c_months4[:, :]

    c_months1 = n.createVariable("c_months1", "S1", ("dim1", "strlen8"))
    c_months1.long_name = "char: One month"
    c_months1[:] = nc.stringtochar(np.array(["December"], dtype="S8"))
    validation['c_months1'] = c_months1[:]

    c_months0 = n.createVariable("c_months0", "S1", ("strlen3",))
    c_months0.long_name = "char: One month (scalar)"
    c_months0[:] = np.array(list("May"))
    validation['c_months0'] = c_months0[:]

    c_numbers = n.createVariable("c_numbers", "S1", ("lat", "lon", "strlen5"))
    c_numbers.long_name = "char: Two dimensional"
    np.empty((2, 3, 5), dtype="S1")
    c_numbers[...] = nc.stringtochar(numbers)
    validation['c_numbers'] = c_numbers[...]

    c_months4m = n.createVariable("c_months4m", "S1", ("time", "strlen7"))
    c_months4m.long_name = "char: Four months (masked)"
    array = nc.stringtochar(months_m)
    c_months4m[:, :] = array
    validation['c_months4m'] = array

    n.close()

    return validation


def test_vlen_string_hdf5(tmp_path):

    #tfile = io.BytesIO()
    our_file = tmp_path/'h5py_vlen.hdf5'
    our_view = tmp_path/'h5py_vlen.txt'
    vlen_strings = ["foo","foobar"]
    make_file_hdf5(our_file, vlen_strings)
    #os.system(f'h5dump {our_file} > {our_view}')
    #with open(our_view,'r') as f:
    #    for line in f.readlines():
    #        print(line)

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

    # Bytes version
    m_array_bytes = [m.encode('utf-8') for m in m_array]

    with nc.Dataset(tfile, 'r') as ncfile:
        ds1 = ncfile.variables['months'][:].tolist()
        assert np.array_equal(m_array, ds1)

    with h5py.File(tfile, 'r') as pfile:
        ds1 = pfile['months'][:].tolist()
        assert np.array_equal(m_array_bytes, ds1)
    
    with pyfive.File(tfile) as hfile:
        ds1 = hfile['months'][:].tolist()
        assert np.array_equal(m_array, ds1)

def test_pathological_strings(tmp_path):
    tfile = tmp_path/'test_strings.nc'
    validation=make_pathological_nc(tfile)
    warnings.warn('Validation of variable length strings assumes h5py is wrong')
    with pyfive.File(tfile) as pfile:
        with h5py.File(tfile) as hfile:
            for k,v in validation.items():
                hdata = hfile[k][...]
                pdata = pfile[k][...]
                try:
                    assert np.array_equal(v, pdata),f'Failed original test for {k}'
                    assert np.array_equal(hdata.astype(str), pdata.astype(str)), f'Failed comparison test for {k}'
                    print(f'--> Passing {k} ({hdata.dtype},{pdata.dtype})')
                except:
                    print(f'---> Failing {k} ({hdata.dtype},{pdata.dtype})')
                    print('Original data', v)
                    print('h5py', hfile[k].dtype, hdata)
                    print('pyfive',pfile[k].dtype, pdata)
                    raise

def test_vlen_contiguous_chunked(tmp_path):
    tfile = tmp_path/'test_strings_2.nc'
    make_contiguous_and_chunked_nc(tfile)

    # Check that slices of the contiguous and chunked vesions of the
    # data are identical. Include slices that span multiple chunks.
    #
    # The array shape is (3, 4) and the chunksize is (2, 2), give four
    # chunks (A, B, C, D) as follows:
    #
    #  +---+---+---+---+
    #  | A | A | B | B |
    #  +---+---+---+---+
    #  | A | A | B | B |
    #  +---+---+---+---+
    #  | C | C | D | D |
    #  +---+---+---+---+
    #
    # So (slice(0, 2), slice(0, 2)) selects the entirety of the A
    # chunks, and no others.

    with pyfive.File(tfile) as pfile:
        contiguous = pfile['months']
        chunked = pfile['months_chunked']

        for index in (
                Ellipsis,
                (slice(1, 3), slice(1, 3)), # spans sub-parts of all 4 chunks
                (1, slice(None)),
                (slice(None), 1),
                (slice(0, 2), slice(0,2)),
        ):
            assert np.array_equal(contiguous[index], chunked[index])
