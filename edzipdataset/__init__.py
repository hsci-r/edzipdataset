from asyncio import AbstractEventLoop
import asyncio
import functools
from io import BytesIO, IOBase
import os
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, TypeVar, Union
from zipfile import ZipInfo, sizeFileHeader # type: ignore
import edzip
import fsspec
import yaml
import sqlite3
from torch.utils.data import Dataset
import yaml
from fsspec.callbacks import TqdmCallback
from s3fs import S3FileSystem
from stream_unzip import stream_unzip
import re
import multiprocessing_utils
from edzipdataset.fsspecutil import SharedMMapCache
import fsspec.asyn

# register out smmap cache
SharedMMapCache.register_cache()
# without this, fsspec hangs in a multiprocessing fork context when it has been already used in the parent process
os.register_at_fork(
    after_in_child=fsspec.asyn.reset_lock,
)


T_co = TypeVar('T_co', covariant=True)

def extract_transform(ezmd: 'EDZipMapDataset', infos: list[Tuple[int,ZipInfo]]) -> list[bytes]:
   return [ezmd.edzip.read(info) for _,info in infos]

def possibly_parallel_extract_transform(ezmd: 'S3HostedEDZipMapDataset', infos: list[Tuple[int,ZipInfo]]) -> list[BytesIO]:
    return ezmd.extract_possibly_in_parallel([info for _,info in infos])

class EDZipMapDataset(Dataset[T_co]):
    """A map dataset class for reading data from a zip file with an external sqlite3 directory."""

    def __init__(self, open_zip: Callable[[],IOBase], open_con: Callable[[],sqlite3.Connection], transform: Callable[['EDZipMapDataset',list[Tuple[int,ZipInfo]]], list[T_co]] = extract_transform, limit: Optional[Sequence[str]] = None):
        """Creates a new instance of the EDZipDataset class.

            Args:
                zip (Callable[[],IOBase]): A function returning a file-like object representing the zip file.
                con (Callable[[],sqlite3.Connection]): A function returning a connection to the SQLite database containing the external directory.
                transform (Callable[['EDZipMapDataset',list[Tuple[int,ZipInfo]]], T_co], optional): A function to transform the zip file entries to the desired output. Defaults to returning a file-like object for the contents.
                limit (Sequence[str]): An optional list of filenames to limit the dataset to.
        """
        self.open_zip = open_zip
        self.open_con = open_con
        self.transform = transform
        self.limit = limit
        self._local = multiprocessing_utils.local()
        

    @property
    def edzip(self) -> edzip.EDZipFile:
        if not hasattr(self._local, 'edzip'):
            self._local.edzip = edzip.EDZipFile(self.open_zip(), self.open_con())
        return self._local.edzip
    
    @property
    def infolist(self) -> Sequence[ZipInfo]:
        if not hasattr(self._local, 'infolist'):
            if self.limit is not None:
                self._local.infolist = list(self.edzip.getinfos(self.limit))
            else:
                self._local.infolist = self.edzip.infolist()
        return self._local.infolist
        
    def __len__(self):
        return len(self.infolist)
    
    def __getitem__(self, idx: int) -> T_co:
        return self.transform(self, [(idx, self.infolist[idx])])[0] # type: ignore

    def __getitems__(self, idxs: list[int]) -> list[T_co]:
        return self.transform(self,zip(idxs,self.edzip.getinfos(idxs))) # type: ignore
    
    def __iter__(self):
        return map(lambda t: self.transform(self, [t])[0], enumerate(self.infolist.__iter__())) # type: ignore
    
    def __setstate__(self, state):
        (
            self.open_zip,
            self.open_con, 
            self.transform, 
            self.limit) = state
        self._local = multiprocessing_utils.local()
    
    def __getstate__(self) -> object:
        return (
            self.open_zip, 
            self.open_con, 
            self.transform, 
            self.limit
        )

def _get_fs(url: str, s3_credentials: dict[str,Optional[str]] | None, asynchronous: bool = False) -> fsspec.AbstractFileSystem:
    """Returns a filesystem configured to use the options given.

    Args:
        url: The URL to use to determine the protocol.
        s3_credentials (dict): possible s3 credentials to use
        block_size (int, optional): The block size to use for the filesystem. Defaults to 5 * 2**20.
        asynchronous (bool, optional): Whether to return an asynchronous filesystem. Defaults to False.

    Returns:
        fs (fsspec.AbstractFileSystem:): The filesystem object.
    """
    protocol = re.match("[^/]+(?=:)",url)
    if protocol is None:
        protocol = ""
    else:
        protocol = protocol.group(0)
    if protocol == "s3" and s3_credentials is not None:
        kwargs = dict(
            key=s3_credentials['aws_access_key_id'] if 'aws_access_key_id' in s3_credentials else None, 
            secret=s3_credentials['aws_secret_access_key'] if 'aws_secret_access_key' in s3_credentials else None, 
            endpoint_url=s3_credentials['endpoint_url'] if 'endpoint_url' in s3_credentials else None
        )
    else:
        kwargs = dict()
    return fsspec.filesystem(protocol, asynchronous=asynchronous, **kwargs)

def derive_sqlite_url_from_zip_url(zipfile_url: str) -> str:
    return zipfile_url + ".offsets.sqlite3"

def derive_sqlite_file_path(sqlite_url: str, sqlite_dir: str) -> str:
    return f"{sqlite_dir}/{os.path.basename(sqlite_url)}"

def ensure_sqlite_database_exists(sqlite_url: str, sqlite_dir: str, s3_credentials: dict[str,Any] | None):
    sqfpath = derive_sqlite_file_path(sqlite_url, sqlite_dir)
    if not os.path.exists(sqfpath):
        os.makedirs(os.path.dirname(sqfpath), exist_ok=True)
        _get_fs(sqlite_url, s3_credentials).get_file(sqlite_url, sqfpath, callback=TqdmCallback(tqdm_kwargs=dict(unit='b', unit_scale=True, dynamic_ncols=True))) # type: ignore

def get_s3_credentials(s3_credentials_yaml_file: Union[str,os.PathLike]) -> dict[str,Optional[str]]:
    with open(s3_credentials_yaml_file, 'r') as f:
        return yaml.safe_load(f)

def _open_s3_zip(zip_url: str, cache_dir: str, s3_credentials: dict[str,Any] | None, block_size: int) -> fsspec.spec.AbstractBufferedFile:
    cache_loc = cache_dir+"/"+os.path.basename(zip_url)
    return _get_fs(zip_url, s3_credentials).open(zip_url, "rb", cache_type="smmap", block_size=block_size, cache_options=dict(location=cache_loc+".cache", index_location=cache_loc+".cache-index"), fill_cache=False) # type: ignore

def _open_file_zip(zip_url: str):
    return open(zip_url, 'rb')

def _open_sqlite(sqlite_file: str):
    return sqlite3.connect(sqlite_file)

class S3HostedEDZipMapDataset(EDZipMapDataset[T_co]):
    """A map dataset class for reading data from an S3 hosted zip file with an external sqlite3 directory."""


    def __init__(self, zip_url:str, cache_dir: str, sqlite_url: Optional[str] = None, s3_credentials_yaml_file: Optional[Union[str,os.PathLike]] = None, block_size: int = 5 * 2**20, transform: Callable[['S3EDZipMapDataset',list[Tuple[int,ZipInfo]]], list[T_co]] = possibly_parallel_extract_transform, *args, **kwargs): # type: ignore
        """Creates a new instance of the S3HostedEDZipDataset class.

            Args:
                zip_url (str): The URL of the zip file on S3.
                sqlite_url (str, optional): The URL of the sqlite3 database file. If not provided, it is derived from the zip_url.
                cache_dir (str): The directory containing the sqlite3 database file and other cached files.
                s3_client (boto3.client): The S3 client object to use.
        """
        if sqlite_url is None:
            sqlite_url = derive_sqlite_url_from_zip_url(zip_url)
        if s3_credentials_yaml_file is not None:
            self.s3_credentials = get_s3_credentials(s3_credentials_yaml_file)
        else:
            self.s3_credentials = None
        if zip_url.startswith('s3:'):
            (self.bucket, self.path) = re.sub('^s3:/?/?', '', zip_url).split('/',1)
            open_zip = functools.partial(_open_s3_zip, zip_url, cache_dir, self.s3_credentials, block_size)
            with open_zip() as f: # This is very hacky. It relies on us having defined open_zip to be _open_s3_zip, which we know returns an S3File with an smmap cache.
                fs: S3FileSystem = f.fs
                acache: SharedMMapCache = open_zip().cache # type: ignore # here we make use of the fact that we know the cache is an smmap cache
                self.acache = acache
                self.acache.afetcher = functools.partial(fs._cat_file, zip_url, None) # and here we make use of the fact that we know it is specifically the S3FileSystem _cat_file
        else:
            self.bucket = None
            self.path = None
            open_zip = functools.partial(_open_file_zip, zip_url)
        ensure_sqlite_database_exists(sqlite_url, cache_dir, self.s3_credentials)
        super().__init__(
            open_zip=open_zip,
            open_con=functools.partial(_open_sqlite,derive_sqlite_file_path(sqlite_url, cache_dir)),
            transform=transform,
            *args, **kwargs)

    async def _a_extract_file(self, offset, size) -> BytesIO:
        compressed_bytes = await self.acache._afetch(offset, offset+size-1)
        _,_, uncompressed_chunks = next(stream_unzip([compressed_bytes]))
        uncompressed_bytes = BytesIO()
        for uncompressed_chunk in uncompressed_chunks:
            uncompressed_bytes.write(uncompressed_chunk)
        uncompressed_bytes.seek(0)
        return uncompressed_bytes
    
    async def _a_extract_files_in_parallel(self, infos: Iterable[ZipInfo], max_extra:int = 128) -> list[BytesIO]:
        return await asyncio.gather(*[self._a_extract_file(zinfo.header_offset, zinfo.compress_size+sizeFileHeader+max_extra) for zinfo in infos])
        
    def extract_possibly_in_parallel(self, infos: list[ZipInfo], max_extra: int = 128) -> list[BytesIO]:
        if len(infos)==1 or self.bucket is None:
            return [BytesIO(self.edzip.read(zinfo)) for zinfo in infos]
        return asyncio.run_coroutine_threadsafe(self._a_extract_files_in_parallel(infos, max_extra), fsspec.asyn.get_loop()).result()
    
    def __getstate__(self):
        return (
            super().__getstate__(), 
            self.s3_credentials,
            self.bucket,
            self.path,
            self.acache
        )

    def __setstate__(self, state):
        (super_state, 
         self.s3_credentials, 
         self.bucket, 
         self.path,
         self.acache) = state
        super().__setstate__(super_state)

