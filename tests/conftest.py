import os
import s3fs
import pathlib
import json
import moto
import pytest

from moto.moto_server.threaded_moto_server import ThreadedMotoServer


# some spoofy server parameters
# test parameters; don't modify these
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
    # the user ID and secret key are needed when accessing a public bucket
    # since our S3 FS and bucket are not actually on an AWS system, they can have
    # bogus values
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

    The S3 FS, being AWS-like but not actually physically deployed anywhere, still needs
    all the usual user IDs, secret keys, endpoint URLs etc; the setup makes use of the ACL=public
    configuration (public-read, or public-read-write). Public DOES NOT mean anon=True, but rather,
    All Users group – https://docs.aws.amazon.com/AmazonS3/latest/userguide/acl-overview.html
    Access permission to this group allows anyone with AWS credentials to access the resource.
    The requests need be signed (authenticated) or not.

    Also, keys are encrypted using AWS-KMS
    https://docs.aws.amazon.com/kms/latest/developerguide/overview.html
    """
    client = get_boto3_client()

    # see not above about ACL=public-read
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
