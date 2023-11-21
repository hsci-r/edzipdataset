import datetime
import mmap
import os
import time
from typing import Any
from fsspec.caching import Fetcher, MMapCache, register_cache, caches

import  logging

class SharedMMapCache(MMapCache):

    name="smmap"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, location: str, index_location: str) -> None:
        super().__init__(blocksize, fetcher, size, location, None)
        self.index_location = index_location
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
    
    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        start_block = start // self.blocksize
        end_block = end // self.blocksize
        need = [i for i in range(start_block, end_block + 1) if self._index[i] != 2]
        while need:
            waiting = []
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
                    self.cache[sstart:send] = self.fetcher(sstart, send)
                    for i in cis:
                        self._index[i] = 2
                elif self._index[i] != 2:
                    waiting.append(i)
            if waiting:
                done = False
                started = datetime.datetime.now()
                while not done and datetime.datetime.now() - started < datetime.timedelta(seconds=30):
                    done = True
                    for block in waiting:
                        if self._index[block] != 2:
                            done = False
                            time.sleep(0.1)
                if not done:
                    for i in waiting:
                        if self._index[i] != 2:
                            self._index[i] = 0
                            need.append(i)
        return self.cache[start:end]
    
    def __getstate__(self):
        state = super().__getstate__()
        del state['_index']

    def __setstate__(self, state):
        super().__setstate__(state)
        self._index = self._makeindex()

    @classmethod
    def register_cache(cls):
        if not cls.name in caches:
            register_cache(cls)
    

register_cache(SharedMMapCache)