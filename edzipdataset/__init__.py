from asyncio import AbstractEventLoop
import asyncio
from io import BytesIO, IOBase
import os
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, TypeVar, Union
from zipfile import ZipInfo, sizeFileHeader # type: ignore
import edzip
import yaml
import sqlite3
from torch.utils.data import Dataset
import yaml
import s3fs
from fsspec.callbacks import TqdmCallback
import aiobotocore.session
from stream_unzip import stream_unzip
import re


T_co = TypeVar('T_co', covariant=True)

def extract_transform(ezmd: 'EDZipMapDataset', infos: list[Tuple[int,ZipInfo]]) -> list[bytes]:
   return [ezmd.edzip.read(info) for _,info in infos]

def parallel_extract_transform(ezmd: 'S3HostedEDZipMapDataset', infos: list[Tuple[int,ZipInfo]]) -> list[BytesIO]:
    return ezmd.extract_in_parallel([info for _,info in infos])

class EDZipMapDataset(Dataset[T_co]):
    """A map dataset class for reading data from a zip file with an external sqlite3 directory."""

    def __init__(self, zip: Callable[...,IOBase], zip_args: list[Any], con: Callable[...,sqlite3.Connection], con_args: list[Any], transform: Callable[['EDZipMapDataset',list[Tuple[int,ZipInfo]]], T_co] = extract_transform, limit: Optional[Sequence[str]] = None):
        """Creates a new instance of the EDZipDataset class.

            Args:
                zip (Callable[...,IOBase]): A function returning a file-like object representing the zip file.
                zip_args (list[Any]): A list of arguments to pass to the zip function.
                con (Callable[...,sqlite3.Connection]): A function returning a connection to the SQLite database containing the external directory.
                con_args (list[Any]): A list of arguments to pass to the con function.
                transform (Callable[['EDZipMapDataset',list[Tuple[int,ZipInfo]]], T_co], optional): A function to transform the zip file entries to the desired output. Defaults to returning a file-like object for the contents.
                limit (Sequence[str]): An optional list of filenames to limit the dataset to.
        """
        self.zip = zip
        self.zip_args = zip_args
        self.con = con
        self.con_args = con_args
        self.transform = transform
        self.limit = limit
        self._edzip = None
        self._infolist = None

    @property
    def edzip(self) -> edzip.EDZipFile:
        if self._edzip is None:
            self._edzip = edzip.EDZipFile(self.zip(*self.zip_args), self.con(*self.con_args))
        return self._edzip
    
    @property
    def infolist(self) -> Sequence[ZipInfo]:
        if self._infolist is None:
            if self.limit is not None:
                self._infolist = list(self.edzip.getinfos(self.limit))
            else:
                self._infolist = self.edzip.infolist()
        return self._infolist
        
    def __len__(self):
        return len(self.infolist)
    
    def __getitem__(self, idx: int) -> T_co:
        return self.transform(self, [(idx, self.infolist[idx])])[0] # type: ignore

    def __getitems__(self, idxs: list[int]) -> list[T_co]:
        return self.transform(self,zip(idxs,self.edzip.getinfos(idxs))) # type: ignore
    
    def __setstate__(self, state):
        (
            self.zip,
            self.zip_args, 
            self.con, 
            self.con_args,
            self.transform, 
            self.limit) = state
        self._edzip = None
        self._infolist = None
    
    def __getstate__(self) -> object:
        return (
            self.zip, 
            self.zip_args,
            self.con, 
            self.con_args,
            self.transform, 
            self.limit
        )

def get_s3_credentials(credentials_yaml_file: Union[str,os.PathLike]) -> dict[str,str]:
    with open(credentials_yaml_file, 'r') as f:
        return yaml.safe_load(f)
    

def get_s3fs(s3_credentials: dict[str,Optional[str]]) -> s3fs.S3FileSystem:
    """Returns an S3 filesystem configured to use the credentials in the provided YAML file.

    Args:
        s3_credentials (dict): possible s3 credentials to use

    Returns:
        s3 (s3fs.S3FileSystem): The S3 client object.
    """
    
    s3 = s3fs.S3FileSystem(
        key=s3_credentials['aws_access_key_id'], 
        secret=s3_credentials['aws_secret_access_key'], 
        endpoint_url=s3_credentials['endpoint_url'])
    return s3

def get_aio_s3_client(s3_credentials: dict[str,Any]) -> aiobotocore.session.ClientCreatorContext:
    return aiobotocore.session.get_session().create_client('s3', **s3_credentials)

def derive_sqlite_url_from_zip_url(zipfile_url: str) -> str:
    return zipfile_url + ".offsets.sqlite3"

def derive_sqlite_file_path(sqlite_url: str, sqlite_dir: str) -> str:
    return f"{sqlite_dir}/{os.path.basename(sqlite_url)}"

def ensure_sqlite_database_exists(sqlite_url: str, sqlite_dir: str, s3_credentials: dict[str,Any] = {}):
    sqfpath = derive_sqlite_file_path(sqlite_url, sqlite_dir)
    if not os.path.exists(sqfpath):
        os.makedirs(os.path.dirname(sqfpath), exist_ok=True)
        get_s3fs(s3_credentials).get_file(sqlite_url, sqfpath, callback=TqdmCallback(tqdm_kwargs=dict(unit='b', unit_scale=True, dynamic_ncols=True))) # type: ignore


def _open_s3_zip(zip_url: str, s3_credentials: dict[str,Any]):
    return get_s3fs(s3_credentials).open(zip_url, fill_cache=False)

def _open_sqlite(sqlite_file: str):
    return sqlite3.connect(sqlite_file)

class S3HostedEDZipMapDataset(EDZipMapDataset[T_co]):
    """A map dataset class for reading data from an S3 hosted zip file with an external sqlite3 directory."""


    def __init__(self, zip_url:str, sqlite_dir: str, sqlite_url: Optional[str] = None, s3_credentials_yaml_file: Optional[Union[str,os.PathLike]] = None, *args, **kwargs):
        """Creates a new instance of the S3HostedEDZipDataset class.

            Args:
                zip_url (str): The URL of the zip file on S3.
                sqlite_url (str, optional): The URL of the sqlite3 database file. If not provided, it is derived from the zip_url.
                sqlite_dir (str): The directory containing the sqlite3 database file.
                s3_client (boto3.client): The S3 client object to use.
        """
        if sqlite_url is None:
            sqlite_url = derive_sqlite_url_from_zip_url(zip_url)
        if s3_credentials_yaml_file is not None:
            self.s3_credentials = get_s3_credentials(s3_credentials_yaml_file)
        else:
            self.s3_credentials = {}
        (self.bucket, self.path) = re.sub('^s3:/?/?', '', zip_url).split('/',1)
        ensure_sqlite_database_exists(sqlite_url, sqlite_dir, self.s3_credentials)
        super().__init__(
            zip=_open_s3_zip,  # type: ignore
            zip_args=[zip_url, self.s3_credentials],
            con=_open_sqlite,
            con_args=[_derive_sqlite_file_path(sqlite_url, sqlite_dir)],
            *args, **kwargs)

    def aio_get_s3_client(self):
        return get_aio_s3_client(s3_credentials=self.s3_credentials)
    
    async def _aio_get_range(self, client, start: int, end: int) -> bytes:
        response = await client.get_object(Bucket=self.bucket, Key=self.path, Range=f"bytes={start}-{end}")
        async with response['Body'] as stream:
            return await stream.read()
        
    async def aio_extract_file(self, client, offset, size) -> BytesIO:
        compressed_bytes = await self._aio_get_range(client, offset, offset+size-1)
        _,_, uncompressed_chunks = next(stream_unzip([compressed_bytes]))
        uncompressed_bytes = BytesIO()
        for uncompressed_chunk in uncompressed_chunks:
            uncompressed_bytes.write(uncompressed_chunk)
        uncompressed_bytes.seek(0)
        return uncompressed_bytes
    
    async def _extract_in_parallel(self, infos: Iterator[ZipInfo], max_extra:int = 128) -> list[BytesIO]:
        async with self.aio_get_s3_client() as client:
            return await asyncio.gather(*[self.aio_extract_file(client, zinfo.header_offset, zinfo.compress_size+sizeFileHeader+max_extra) for zinfo in infos])
        
    def extract_in_parallel(self, infos: Iterator[ZipInfo], max_extra: int = 128, loop: Optional[AbstractEventLoop] = None) -> list[BytesIO]:
        if loop is None and asyncio._get_running_loop() is not None:
            loop = asyncio.get_running_loop()
        if loop is not None:
            return asyncio.run_coroutine_threadsafe(self._extract_in_parallel(infos, max_extra), loop).result()
        return asyncio.run(self._extract_in_parallel(infos))
