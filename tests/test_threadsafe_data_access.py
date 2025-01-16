import os
import s3fs
import pathlib
import pyfive
import pytest
import h5netcdf
import netCDF4
import numpy as np
import dask.array as da

# needed by the spoofed s3 filesystem
port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


def _get_s3_file(s3fs_s3, ncfile):
    """Copy a POSIX file to S3."""
    # set up physical file and Path properties
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name

    # use mocked s3fs
    bucket = "MY_BUCKET"
    try:
        s3fs_s3.mkdir(bucket)
    except FileExistsError:
        # Bucket was already created by another test
        pass

    s3fs_s3.put(file_path, bucket)
    s3 = s3fs.S3FileSystem(
        anon=False,
        version_aware=True,
        client_kwargs={"endpoint_url": endpoint_uri}
    )
    return s3.open(os.path.join("MY_BUCKET", file_name), "rb")

                 
def test_threadsafe_datea_accesss(s3fs_s3):
    """test that the data is correctly retrieved using concurrent threads"""
    # Create a chunked POSIX file 
    chunked_file = "./threading_test_file_16_chunks.nc"
    
    n = netCDF4.Dataset(chunked_file, 'w')
    chunksizes=(6, 32, 32)
    z = n.createDimension('z', 12)
    y = n.createDimension('y', 64)
    x = n.createDimension('x', 128)
    
    v = n.createVariable('v', 'f8',
                         dimensions=['z', 'y', 'x'],
                         fill_value=False,
                         contiguous=False,
                         chunksizes=(6, 32, 32),
                         complevel=1)
    v[...] = np.random.random(v.size).reshape(12, 64, 128)
    n.close()

    posix = chunked_file

    # Get s3 access
    s3 = _get_s3_file(s3fs_s3, chunked_file)

    # Set number of iterations for checking thread safety. Need to be
    # quite large to avoid cases where the code is not threadsafe, but
    # still retrieves the correct data by good fortune.
    n_iterations = 1000
    
    for ftype, filename in zip(
            ('POSIX', 'S3'),
            (posix, s3),
    ):
        print (f"\n{ftype} ----------------\n")

        # Get the file data array, on asingle thread
        with pyfive.File(chunked_file) as hf:
            v = hf['v']
        print (v)
        print (f"Storage chunk size: {v.chunks}")
        array0 = v[...].view(type=np.ndarray)

        # Loop round different Dask chunk patterns. These are designed
        # to various coincide and not coincide with the HDF5 chunks in
        # the file.
        for chunks in (
                v.shape,
                v.chunks,
                (12, 32, 32),
                (11, 63, 127),
                (6, 20, 20)
        ):
            dx = da.from_array(v, chunks=chunks)
            print (f"\n{dx.npartitions} Dask chunks: {dx.chunks}")
            
            for i in range(n_iterations):
                # Use Dask to get the array with one or more threads
                try:
                    array = dx.compute()
                except Exception as error:
                    print (f"Failed on iteration {i + 1}")
                    raise
                else:
                    # Compare the array created with multiple threads
                    # with that created with one thread
                    if not (array == array0).all():
                        print (f"Failed on iteration {i + 1}")
                        raise ValueError(
                            "At least one Dask chunk read at least one wrong "
                            "value from the file (likely from parts of "
                            "storage chunks that it should not have been "
                            "accessing, due to conflicting seeks on the same "
                            "open file handle). "
                            f"Storage chunks: {v.chunks}, "
                            f"Dask chunks: {dx.chunks}"
                        )
            else:
                print (f"Completed {n_iterations} iterations")

    # Tidy up
    os.remove(chunked_file)
