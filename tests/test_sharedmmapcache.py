from multiprocessing import Process
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, call

import fsspec
from fsspec.implementations.http import HTTPFileSystem
from edzipdataset.fsspecutil import SharedMMapCache


def paral_fetch(fetcher, cache_dir):
    c = SharedMMapCache(blocksize=1024, fetcher=fetcher, size=65536, location=cache_dir+"/cache", index_location=cache_dir+"/cache-index")
    c._fetch(0,256)
    c._fetch(45,600)
    c._fetch(2100,4100)
    c._fetch(2200,4200)

def _paral_fetcher(start, end) -> bytes:
    ret = bytearray(end-start)
    for i in range(start, end):
        ret[i-start] = i % 256
    time.sleep(0.5)
    return bytes(ret)

paral_fetcher = Mock(side_effect=_paral_fetcher)


class TestSharedMMapCache(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
            
    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_sharedmmapcache(self):
        def _fetcher(start, end) -> bytes:
            ret = bytearray(end-start)
            for i in range(start, end):
                ret[i-start] = i % 256
            return bytes(ret)
        fetcher = Mock(side_effect=_fetcher)
        c = SharedMMapCache(blocksize=1024, fetcher=fetcher, size=65536, location=self.dir+"/cache", index_location=self.dir+"/cache-index")
        self.assertEqual(c._fetch(0,256), bytes([n % 256 for n in range(0,256)]))
        self.assertEqual(c._fetch(45,600), bytes([n % 256 for n in range(45,600)]))
        fetcher.assert_called_once_with(0,1024)
        self.assertEqual(c._fetch(2100,4100), bytes([n % 256 for n in range(2100,4100)]))
        self.assertEqual(c._fetch(2200,4200), bytes([n % 256 for n in range(2200,4200)]))
        fetcher.assert_called_with(2048,5120)
        self.assertEqual(len(fetcher.mock_calls), 2)


    def test_sharedmmapcache_multiprocessing(self):
        p = Process(target=paral_fetch, args=(self.dir,))
        p.start()
        c = SharedMMapCache(blocksize=1024, fetcher=paral_fetcher, size=65536, location=self.dir+"/cache2", index_location=self.dir+"/cache-index2")
        time.sleep(0.6)
        self.assertEqual(c._fetch(0,256), bytes([n % 256 for n in range(0,256)]))
        self.assertEqual(c._fetch(45,600), bytes([n % 256 for n in range(45,600)]))
        self.assertEqual(c._fetch(2100,4100), bytes([n % 256 for n in range(2100,4100)]))
        self.assertEqual(c._fetch(2200,4200), bytes([n % 256 for n in range(2200,4200)]))
        p.join()
        paral_fetcher.assert_has_calls([call(0,1024),call(2048,5120)])
        self.assertEqual(len(paral_fetcher.mock_calls), 2)

    def test_sharedmmapcache_loading(self):
        self.assertRaises(TypeError, lambda: HTTPFileSystem().open("https://www.google.com/robots.txt", "rb", cache_type="smmap"))
        with fsspec.open("https://www.google.com/robots.txt", "rb", cache_type="smmap", cache_options=dict(location=self.dir+"/cache-2", index_location=self.dir+"/cache-index-2")) as f:
            self.assertIsInstance(f.cache, SharedMMapCache) # type: ignore

