from pathlib import Path
import fsspec.asyn

import asyncio
from hscitorchutil.fsspec import get_async_filesystem, prefetch_if_remote, _get_afetcher, multi_fetch_async
from fsspec.asyn import AsyncFileSystem
import concurrent.futures


def test_local_async():
    assert isinstance(get_async_filesystem("/"), AsyncFileSystem)


def test_prefetch_if_remote(tmp_path: Path):
    with open(tmp_path / "test.txt", 'w') as f:
        f.write("hello")
    assert prefetch_if_remote(str(tmp_path / "test.txt"),
                              5, cache_dir=str(tmp_path / "cache")).done() == True
    f = prefetch_if_remote("http://google.com/", 50,
                           cache_dir=str(tmp_path / "cache"))
    concurrent.futures.wait([f], timeout=10)
    assert f.done()


def test_afetcher(tmp_path: Path):
    with open(tmp_path / "test.txt", 'w') as f:
        f.write("hello")
    afetcher = _get_afetcher(str(tmp_path / "test.txt"),
                             5, cache_dir=str(tmp_path / "cache"))
    f = asyncio.run_coroutine_threadsafe(
        afetcher(0, 50), fsspec.asyn.get_loop())
    concurrent.futures.wait([f], timeout=10)
    assert f.done()
    assert f.result() == b"hello"
    afetcher = _get_afetcher("http://google.com/", 2 **
                             17, cache_dir=str(tmp_path / "cache"))
    f = asyncio.run_coroutine_threadsafe(
        afetcher(0, 50), fsspec.asyn.get_loop())
    concurrent.futures.wait([f], timeout=10)
    assert f.done()


def test_multi_fetch_async(tmp_path: Path):
    with open(tmp_path / "test1.txt", 'w') as f:
        f.write("hello"*100)
    with open(tmp_path / "test2.txt", 'w') as f:
        f.write("world"*100)
    afetcher1 = _get_afetcher(
        str(tmp_path / "test1.txt"), 500, cache_dir=str(tmp_path / "cache"))
    afetcher2 = _get_afetcher(
        str(tmp_path / "test2.txt"), 500, cache_dir=str(tmp_path / "cache"))
    assert multi_fetch_async([(afetcher1, [(0, 5), (4, 9)]), (afetcher2, [
                             (0, 5), (4, 9)])]) == [[b"hello", b"ohell"], [b"world", b"dworl"]]
