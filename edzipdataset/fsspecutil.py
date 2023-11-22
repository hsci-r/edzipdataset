import asyncio
import datetime
import mmap
import os
import time
from typing import Any, Awaitable, Callable, Tuple
from fsspec.caching import Fetcher, MMapCache, register_cache, caches

class SharedMMapCache(MMapCache):

    name="smmap"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, location: str, index_location: str, afetcher: Callable[[int, int], Awaitable[bytes]] | None = None, parallel_timeout = 30) -> None:
        super().__init__(blocksize, fetcher, size, location, None)
        self.index_location = index_location
        self.parallel_timeout = parallel_timeout
        self.afetcher = afetcher
        self._index = self._makeindex()

    def _makeindex(self) -> mmap.mmap | bytearray:
        if self.size == 0:
            return bytearray()
        if not os.path.exists(self.index_location):
            fd = open(self.index_location, "wb+")
            fd.seek(self.size // self.blocksize)
            fd.write(b'\x00')
            fd.flush()
        else:
            fd = open(self.index_location, "rb+")
        return mmap.mmap(fd.fileno(), self.size // self.blocksize + 1)
    
    def _get_need(self, start: int | None, end: int | None) -> list[int]:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        start_block = start // self.blocksize
        end_block = end // self.blocksize
        return [i for i in range(start_block, end_block + 1) if self._index[i] != 2]
    
    def _wait(self, waiting: list[int], need: list[int]):
        if waiting:
            done = False
            started = datetime.datetime.now()
            while not done and datetime.datetime.now() - started < datetime.timedelta(seconds=30):
                done = True
                for block in waiting:
                    if self._index[block] != 2:
                        done = False
                        time.sleep(0.1)
            if not done: # Waited for 30 seconds for other processes to finish fetching the needed blocks. Give up and do it ourselves.
                for i in waiting:
                    if self._index[i] != 2:
                        self._index[i] = 0
                        need.append(i)

    async def _await(self, waiting: list[int], need: list[int]):
        if waiting:
            done = False
            started = datetime.datetime.now()
            while not done and datetime.datetime.now() - started < datetime.timedelta(seconds=30):
                done = True
                for block in waiting:
                    if self._index[block] != 2:
                        done = False
                        await asyncio.sleep(0.1)
            if not done: # Waited for 30 seconds for other processes to finish fetching the needed blocks. Give up and do it ourselves.
                for i in waiting:
                    if self._index[i] != 2:
                        self._index[i] = 0
                        need.append(i)

    def _get_to_fetch(self, need: list[int]) -> Tuple[list[Tuple[int,int,list[int]]], list[int]]:
        waiting: list[int] = []
        to_fetch: list[Tuple[int,int,list[int]]] = []
        while need:
            i = need.pop(0)
            if self._index[i] == 0:
                self._index[i] = 1
                sstart = i * self.blocksize
                cis = [i]
                while need and need[0] == i+1 and self._index[need[0]] == 0:
                    i = need.pop(0)
                    self._index[i] = 1
                    cis.append(i)
                send = min(i * self.blocksize + self.blocksize, self.size)
                to_fetch.append((sstart, send, cis))
            elif self._index[i] != 2:
                waiting.append(i)
        return (to_fetch, waiting)

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        need = self._get_need(start, end)
        while need:
            to_fetch, waiting = self._get_to_fetch(need)
            for sstart, send, cis in to_fetch:
                self.cache[sstart:send] = self.fetcher(sstart, send)
                for i in cis:
                    self._index[i] = 2
            self._wait(waiting, need)
        return self.cache[start:end]
    
    async def _afetch(self, start: int | None, end: int | None) -> bytes:
        need = self._get_need(start, end)
        while need:
            to_fetch, waiting = self._get_to_fetch(need)
            datas = await asyncio.gather(*[self.afetcher(sstart, send) for sstart, send, _ in to_fetch]) # type: ignore
            for (sstart, send, cis), data in zip(to_fetch, datas):
                self.cache[sstart:send] = data
                for i in cis:
                    self._index[i] = 2
            await self._await(waiting, need)
        return self.cache[start:end]

    
    def __getstate__(self):
        state = super().__getstate__()
        del state['_index']
        return state

    def __setstate__(self, state: dict[str, Any]):
        super().__setstate__(state)
        self._index = self._makeindex()

    @classmethod
    def register_cache(cls):
        if not cls.name in caches:
            register_cache(cls)
    

register_cache(SharedMMapCache)