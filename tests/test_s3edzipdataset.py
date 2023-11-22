import asyncio
from io import BytesIO, IOBase
import json
import os
import pickle
import shutil
import tempfile
from typing import Tuple
from unittest.mock import Mock
from zipfile import ZipFile, ZipInfo

from edzip.sqlite import create_sqlite_directory_from_zip
from edzipdataset import S3HostedEDZipMapDataset, possibly_parallel_extract_transform
import pytest
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from edzipdataset.dsutil import DatasetToIterableDataset

port = 5555
endpoint_url = "http://127.0.0.1:%s/" % port

def get_boto3_client():
    from botocore.session import Session

    # NB: we use the sync botocore client for setup
    session = Session()
    return session.create_client("s3", endpoint_url=endpoint_url)



@pytest.fixture(scope="module")
def tmpdir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)

@pytest.fixture(scope="module")
def s3(tmpdir):
    # writable local S3 system

    # This fixture is module-scoped, meaning that we can re-use the MotoServer across all tests
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    zfbuffer = BytesIO()
    with ZipFile(zfbuffer, "w") as zf:
            zf.writestr("test.txt", "Hello, world!")
            zf.writestr("test2.txt", "Hello again!")
            zf.writestr("test3.txt", "Goodbye!")
            create_sqlite_directory_from_zip(zf, tmpdir+"/test.zip.offsets.sqlite3").close()    
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"            
    with open(tmpdir+"/s3_secret.yaml", "w") as f:
        f.write("aws_access_key_id: foo\n")
        f.write("aws_secret_access_key: foo\n")
        f.write("endpoint_url: "+endpoint_url+"\n")
    client = get_boto3_client()
    client.create_bucket(Bucket="test", ACL="public-read")
    client.put_object(Bucket="test", Key="test.zip", Body=zfbuffer.getvalue())
    yield
    server.stop()    

def _transform(edmd: S3HostedEDZipMapDataset, infos:list[Tuple[int,ZipInfo]]): 
    return [edmd.edzip.open(zinfo).read()+str(idx).encode() for idx,zinfo in infos]

def test_s3hostededzipdataset(tmpdir, s3):
    ds = S3HostedEDZipMapDataset[IOBase]("s3://test/test.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml", transform=_transform)
    assert len(ds) == 3
    assert ds[0] == b"Hello, world!0"
    assert ds[1] == b"Hello again!1"
    assert ds[2] == b"Goodbye!2"
    assert ds.__getitems__([0,2]) == [b"Hello, world!0", b"Goodbye!2"]

def test_s3hostededzipdataset_limit(tmpdir, s3):
    ds = S3HostedEDZipMapDataset[IOBase]("s3://test/test.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml",transform=_transform, limit=["test.txt", "test3.txt"])
    assert len(ds) == 2
    assert ds[0] == b"Hello, world!0"
    assert ds[1] == b"Goodbye!1"

def test_pickling_s3hosteedzipdataset(tmpdir, s3):
    ds = S3HostedEDZipMapDataset[IOBase]("s3://test/test.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml", transform=_transform, limit=["test.txt", "test3.txt"])
    assert len(ds) == 2
    assert ds[0] == b"Hello, world!0"
    assert ds[1] == b"Goodbye!1"
    bytes = pickle.dumps(ds)
    ds2 = pickle.loads(bytes)
    assert len(ds2) == 2
    assert ds2[0] == b"Hello, world!0"
    assert ds2[1] == b"Goodbye!1"

def test_s3hostededzipdataset_as_iterabledataset(tmpdir, s3):
    ds = DatasetToIterableDataset(S3HostedEDZipMapDataset[IOBase]("s3://test/test.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml", transform=_transform, limit=["test.txt", "test3.txt"]))
    expected = [b"Hello, world!0",b"Goodbye!1" ]
    for act, exp in zip(ds,expected):
        assert act == exp



def test_s3hostededzipdataset_async(tmpdir, s3, monkeypatch):
    transform = Mock(side_effect=possibly_parallel_extract_transform)
    mrun = Mock(wraps=asyncio.run)
    monkeypatch.setattr(asyncio, "run", mrun)
    ds = S3HostedEDZipMapDataset[BytesIO]("s3://test/test.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml", transform=transform)
    assert len(ds) == 3
    assert ds[0].getvalue() == b"Hello, world!"
    assert ds[1].getvalue() == b"Hello again!"
    assert ds[2].getvalue() == b"Goodbye!"
    assert transform.call_count == 3
    mrun.assert_not_called()
    assert list(map(lambda x: x.getvalue(), ds.__getitems__([0,1,2]))) == [b"Hello, world!", b"Hello again!", b"Goodbye!"]
    assert transform.call_count == 4
    mrun.assert_called_once()
