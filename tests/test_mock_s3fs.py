import os
import s3fs
import pathlib
import pyfive
import pytest
import h5netcdf


# needed by the spoofed s3 filesystem
port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


def test_s3fs_s3(s3fs_s3):
    """Test mock S3 filesystem constructor."""
    # this is an entire mock S3 FS
    mock_s3_filesystem = s3fs_s3

    # explore its attributes and methods
    print(dir(mock_s3_filesystem))

    assert not mock_s3_filesystem.anon
    assert not mock_s3_filesystem.version_aware
    assert mock_s3_filesystem.client_kwargs == {'endpoint_url': 'http://127.0.0.1:5555/'}


def test_s3file_with_s3fs(s3fs_s3):
    """
    This test spoofs a complete s3fs FileSystem via s3fs_s3,
    creates a mock bucket inside it, then puts a REAL netCDF4 file in it,
    then it loads it as if it was an S3 file. This is proper
    Wild Weasel stuff right here.
    """
    # set up physical file and Path properties
    ncfile = "./tests/data/issue23_A.nc"
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name

    # use mocked s3fs
    bucket = "MY_BUCKET"
    s3fs_s3.mkdir(bucket)
    s3fs_s3.put(file_path, bucket)
    s3 = s3fs.S3FileSystem(
        anon=False, version_aware=True, client_kwargs={"endpoint_url": endpoint_uri}
    )

    # test load by h5netcdf
    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        print("File path", f.path)
        ncfile = h5netcdf.File(f, 'r', invalid_netcdf=True)
        print("File loaded from spoof S3 with h5netcdf:", ncfile)
        print(ncfile["q"])
    assert "q" in ncfile

    # PyFive it
    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        pyfive_ds = pyfive.File(f)
        print(f"Dataset loaded from mock S3 with s3fs and Pyfive: ds")
        assert "q" in pyfive_ds
