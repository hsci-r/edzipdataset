from io import BytesIO, IOBase
import tempfile
import unittest
from unittest.mock import patch

import shutil
from zipfile import ZipFile, ZipInfo
from edzip import EDZipFile, create_sqlite_directory_from_zip
import pickle

from edzipdataset import S3HostedEDZipMapDataset

def _transform(edzip: EDZipFile,idx: int,zinfo: ZipInfo): 
    return edzip.open(zinfo).read()+str(idx).encode()

class TestS3HostedEDZipMapDataset(unittest.TestCase):
    def setUp(self):
        zfbuffer = BytesIO()
        zfbuffer.name = "test.zip"
        self.tmpdir = tempfile.mkdtemp()
        with ZipFile(zfbuffer, "w") as zf:
            zf.writestr("test.txt", "Hello, world!")
            zf.writestr("test2.txt", "Hello again!")
            zf.writestr("test3.txt", "Goodbye!")
            create_sqlite_directory_from_zip(zf, self.tmpdir+"/test.zip.offsets.sqlite3_orig").close()
        sqf = open(self.tmpdir+"/test.zip.offsets.sqlite3_orig", "rb")
        def se(url,*args,**kwargs): 
            if url == "s3://foo/test.zip":
                return zfbuffer
            else:
                return sqf
        self.patchers = [
            patch("smart_open.open", side_effect=se),
            patch("edzipdataset.get_s3_client", return_value=None),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_s3hostededzipdataset(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, s3_credentials_yaml_file="")
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].read(), b"Hello, world!")
        self.assertEqual(ds[1].read(), b"Hello again!")
        self.assertEqual(ds[2].read(), b"Goodbye!")
        self.assertEqual(list(map(lambda x: x.read(), ds.__getitems__([0,2]))), [b"Hello, world!", b"Goodbye!"])

    def test_s3hostededzipdataset_limit(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, s3_credentials_yaml_file="", limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0].read(), b"Hello, world!")
        self.assertEqual(ds[1].read(), b"Goodbye!")

    def test_s3hostededzipdataset_transform(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, s3_credentials_yaml_file="", transform=_transform, limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0], b"Hello, world!0")
        self.assertEqual(ds[1], b"Goodbye!1")

    def test_pickling_s3hosteedzipdataset(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, s3_credentials_yaml_file="", transform=_transform, limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0], b"Hello, world!0")
        self.assertEqual(ds[1], b"Goodbye!1")
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 2)
        self.assertEqual(ds2[0], b"Hello, world!0")
        self.assertEqual(ds2[1], b"Goodbye!1")

