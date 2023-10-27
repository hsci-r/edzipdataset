from io import BytesIO, IOBase
import tempfile
import unittest
from unittest.mock import patch

import shutil
from zipfile import ZipFile
from edzip import create_sqlite_directory_from_zip

from edzipdataset import S3HostedEDZipMapDataset
class TestS3HostedEDZipMapDataset(unittest.TestCase):
    def setUp(self):
        zfbuffer = BytesIO()
        zfbuffer.name = "test.zip"
        self.tmpdir = tempfile.mkdtemp()
        with ZipFile(zfbuffer, "w") as zf:
            zf.writestr("test.txt", "Hello, world!")
            zf.writestr("test2.txt", "Hello again!")
            zf.writestr("test3.txt", "Goodbye!")
            create_sqlite_directory_from_zip(zf, self.tmpdir+"/test.zip.offsets.sqlite3")
        self.patcher = patch("smart_open.open", return_value=zfbuffer)
        self.patcher.start()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        self.patcher.stop()

    def test_s3hostededzipdataset(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].read(), b"Hello, world!")
        self.assertEqual(ds[1].read(), b"Hello again!")
        self.assertEqual(ds[2].read(), b"Goodbye!")
        self.assertEqual(list(map(lambda x: x.read(), ds.__getitems__([0,2]))), [b"Hello, world!", b"Goodbye!"])

    def test_s3hostededzipdataset_limit(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0].read(), b"Hello, world!")
        self.assertEqual(ds[1].read(), b"Goodbye!")

    def test_s3hostededzipdataset_transform(self):
        ds = S3HostedEDZipMapDataset[IOBase]("s3://foo/test.zip", self.tmpdir, transform=lambda edzip,idx,zinfo: edzip.open(zinfo).read()+str(idx).encode(), limit=["test.txt", "test3.txt"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0], b"Hello, world!0")
        self.assertEqual(ds[1], b"Goodbye!1")
