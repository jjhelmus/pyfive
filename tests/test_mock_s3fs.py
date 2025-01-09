import os
import s3fs
import pathlib
import json
import moto
import pyfive
import pytest

import h5netcdf

from moto.moto_server.threaded_moto_server import ThreadedMotoServer


# some spoofy server parameters
port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port
test_bucket_name = "test"
versioned_bucket_name = "test-versioned"
secure_bucket_name = "test-secure"

def get_boto3_client():
    from botocore.session import Session

    # NB: we use the sync botocore client for setup
    session = Session()
    return session.create_client("s3", endpoint_url=endpoint_uri)

@pytest.fixture(scope="module")
def s3_base():
    # writable local S3 system

    # This fixture is module-scoped, meaning that we can re-use the MotoServer across all tests
    #####
    # lifted from https://github.com/fsspec/s3fs/blob/main/s3fs/tests/test_s3fs.py
    #####
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    os.environ.pop("AWS_PROFILE", None)

    print("server up")
    yield
    print("moto done")
    server.stop()


@pytest.fixture()
def s3fs_s3(s3_base):
    """
    Create a fully functional "virtual" S3 FileSystem compatible with fsspec/s3fs.
    Method inspired by https://github.com/fsspec/s3fs/blob/main/s3fs/tests/test_s3fs.py
    """
    client = get_boto3_client()
    client.create_bucket(Bucket=test_bucket_name, ACL="public-read")

    client.create_bucket(Bucket=versioned_bucket_name, ACL="public-read")
    client.put_bucket_versioning(
        Bucket=versioned_bucket_name, VersioningConfiguration={"Status": "Enabled"}
    )

    # initialize secure bucket
    client.create_bucket(Bucket=secure_bucket_name, ACL="public-read")
    policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Id": "PutObjPolicy",
            "Statement": [
                {
                    "Sid": "DenyUnEncryptedObjectUploads",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": "s3:PutObject",
                    "Resource": "arn:aws:s3:::{bucket_name}/*".format(
                        bucket_name=secure_bucket_name
                    ),
                    "Condition": {
                        "StringNotEquals": {
                            "s3:x-amz-server-side-encryption": "aws:kms"
                        }
                    },
                }
            ],
        }
    )

    client.put_bucket_policy(Bucket=secure_bucket_name, Policy=policy)
    s3fs.S3FileSystem.clear_instance_cache()
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": endpoint_uri})
    s3.invalidate_cache()

    yield s3


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
