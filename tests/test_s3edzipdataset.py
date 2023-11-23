import asyncio
from io import BytesIO, IOBase
import json
import os
import pickle
import shutil
import tempfile
import time
from typing import Tuple
from unittest.mock import Mock, call
from zipfile import ZipFile, ZipInfo

from edzip.sqlite import create_sqlite_directory_from_zip
from edzipdataset import S3HostedEDZipMapDataset, possibly_parallel_extract_transform
import pytest
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from edzipdataset.dsutil import DatasetToIterableDataset
from multiprocessing import Process, set_start_method

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
    zfbuffer = BytesIO()
    with ZipFile(zfbuffer, "w") as zf:
            for i in range(500000):
                zf.writestr(f"test{i}.txt", f"{i}")
            create_sqlite_directory_from_zip(zf, tmpdir+"/large.zip.offsets.sqlite3").close()    
    client.put_object(Bucket="test", Key="large.zip", Body=zfbuffer.getvalue())
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



def test_s3hostededzipdataset_async(tmpdir, s3):
    transform = Mock(side_effect=possibly_parallel_extract_transform)
    ds = S3HostedEDZipMapDataset[BytesIO]("s3://test/large.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml", block_size=1024, transform=transform)
    assert len(ds) == 500000
    assert ds[0].getvalue() == b"0"
    assert ds[1].getvalue() == b"1"
    assert ds[2].getvalue() == b"2"
    assert transform.call_count == 3
    assert list(map(lambda x: x.getvalue(), ds.__getitems__([0,430,20000,430000]))) == [b"0", b"430", b"20000", b"430000"]
    assert transform.call_count == 4


def _process(ds, i, sleep):
    ds.__getitems__([i,430*i,20000+i,430000-5*i])
    time.sleep(sleep)
    assert list(map(lambda x: x.getvalue(), ds.__getitems__([50,20100,430100]))) == [b"50", b"20100", b"430100"]

set_start_method('fork')


def test_s3hostededzipdataset_async_multiprocess(tmpdir, s3):
    ds = S3HostedEDZipMapDataset[BytesIO]("s3://test/large.zip", tmpdir, s3_credentials_yaml_file=tmpdir+"/s3_secret.yaml", block_size=1024, transform=possibly_parallel_extract_transform)
    ps = [Process(target=_process, args=(ds, i, i/8)) for i in range(20)]
    for p in ps:
        p.start()
    _process(ds, 25, 0.5)
    for p in ps:
        p.join()
