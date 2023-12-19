from pathlib import Path
import fsspec.asyn

import asyncio
from hscitorchutil.fsspec import get_async_filesystem, prefetch_if_remote, _get_afetcher
from fsspec.asyn import AsyncFileSystem
import concurrent.futures


def test_local_async():
    assert isinstance(get_async_filesystem("/"), AsyncFileSystem)


def test_prefetch_if_remote(tmp_path: Path):
    with open(tmp_path / "test.txt", 'w') as f:
        f.write("hello")
    assert prefetch_if_remote(str(tmp_path / "test.txt"), 5, cache_dir=str(tmp_path / "cache")).done() == True
    f = prefetch_if_remote("http://google.com/", 50, cache_dir=str(tmp_path / "cache"))
    concurrent.futures.wait([f], timeout=10)
    assert f.done()


def test_afetcher(tmp_path: Path):
    with open(tmp_path / "test.txt", 'w') as f:
        f.write("hello")
    afetcher = _get_afetcher(str(tmp_path / "test.txt"), 5, cache_dir=str(tmp_path / "cache"))
    f = asyncio.run_coroutine_threadsafe(afetcher(0,50), fsspec.asyn.get_loop())
    concurrent.futures.wait([f], timeout=10)
    assert f.done()
    assert f.result() == b"hello"
    afetcher = _get_afetcher("http://google.com/", 50, cache_dir=str(tmp_path / "cache"))
    f = asyncio.run_coroutine_threadsafe(afetcher(0,50), fsspec.asyn.get_loop())
    concurrent.futures.wait([f], timeout=10)
    assert f.done()
    