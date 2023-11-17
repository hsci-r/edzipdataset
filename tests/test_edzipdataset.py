from io import BytesIO, IOBase
import tempfile
from typing import Tuple
import unittest
from unittest.mock import Mock, patch

import shutil
from zipfile import ZipFile, ZipInfo
from edzip.sqlite import create_sqlite_directory_from_zip
import pickle
from fsspec import AbstractFileSystem

from edzipdataset import S3HostedEDZipMapDataset
from edzipdataset.dsutil import DatasetToIterableDataset

def _transform(edmd: S3HostedEDZipMapDataset, infos:list[Tuple[int,ZipInfo]]): 
    return [edmd.edzip.open(zinfo).read()+str(idx).encode() for idx,zinfo in infos]

class TestS3HostedEDZipMapDataset(unittest.TestCase):
    def setUp(self):
        zfbuffer = BytesIO()
        zfbuffer.name = "test.zip"
        self.tmpdir = tempfile.mkdtemp()
        with ZipFile(zfbuffer, "w") as zf:
            zf.writestr("test.txt", "Hello, world!")
            zf.writestr("test2.txt", "Hello again!")
            zf.writestr("test3.txt", "Goodbye!")
            create_sqlite_directory_from_zip(zf, self.tmpdir+"/test.zip.offsets.sqlite3").close()
        fs = Mock(spec=AbstractFileSystem)
        fs.open.return_value = zfbuffer
        self.patchers = [
            patch("edzipdataset.get_fs", return_value=fs),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_s3hostededzipdataset(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, transform=_transform)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], b"Hello, world!0")
        self.assertEqual(ds[1], b"Hello again!1")
        self.assertEqual(ds[2], b"Goodbye!2")
        self.assertEqual(ds.__getitems__([0,2]), [b"Hello, world!0", b"Goodbye!2"])

    def test_s3hostededzipdataset_limit(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, transform=_transform, limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0], b"Hello, world!0")
        self.assertEqual(ds[1], b"Goodbye!1")

    def test_pickling_s3hosteedzipdataset(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, transform=_transform, limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0], b"Hello, world!0")
        self.assertEqual(ds[1], b"Goodbye!1")
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 2)
        self.assertEqual(ds2[0], b"Hello, world!0")
        self.assertEqual(ds2[1], b"Goodbye!1")

    def test_s3hostededzipdataset_as_iterabledataset(self):
        ds = DatasetToIterableDataset(S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, transform=_transform, limit=["test.txt", "test3.txt"]))
        expected = [b"Hello, world!0",b"Goodbye!1" ]
        for act, exp in zip(ds,expected):
            self.assertEqual(act, exp)


