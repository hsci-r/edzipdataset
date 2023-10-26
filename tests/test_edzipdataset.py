from io import BytesIO
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import shutil
from zipfile import ZipFile
from edzip import create_sqlite_directory_from_zip
from hereutil import here

from edzipdataset import S3HostedEDZipDataset

def mock_responses(responses, default_response=None):
  return lambda input: responses[input] if input in responses else default_response

class TestS3HostedEDZipDataset(unittest.TestCase):
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
        ds = S3HostedEDZipDataset("s3://foo/test.zip", self.tmpdir)
        self.assertIsInstance(ds, S3HostedEDZipDataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].read(), b"Hello, world!")
        self.assertEqual(ds[1].read(), b"Hello again!")
        self.assertEqual(ds[2].read(), b"Goodbye!")
        self.assertEqual(list(map(lambda x: x.read(), ds.__getitems__([0,2]))), [b"Hello, world!", b"Goodbye!"])
