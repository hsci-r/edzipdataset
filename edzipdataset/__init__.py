from io import IOBase
import os
from typing import Any, Callable, Generic, Sequence, TypeVar, Union
from zipfile import ZipInfo
import edzip
import smart_open
import yaml
import boto3
import sqlite3
from torch.utils.data import Dataset
import shutil
import yaml
import boto3

def get_s3_client(credentials_yaml_file: Union[str,os.PathLike]):
    """Returns an S3 client configured to use the credentials in the provided YAML file.

    Args:
        credentials_yaml_file (str): The path to the YAML file containing the AWS credentials.

    Returns:
        s3_client (boto3.client): The S3 client object.
    """
    with open(credentials_yaml_file, 'r') as f:
        credentials = yaml.safe_load(f)
    session = boto3.Session()
    s3_client = session.client(service_name='s3', **credentials)
    print(credentials)
    return s3_client


T = TypeVar('T')

class EDZipDataset(Dataset[T]):
    """A dataset class for reading data from a zip file with an external sqlite3 directory."""

    def __init__(self, zip: IOBase, con: sqlite3.Connection, transform: Callable[[edzip.EDZipFile,int,ZipInfo], T] = lambda edzip,idx,zinfo: edzip.open(zinfo), limit: Union[Sequence[str],None] = None):
        """Creates a new instance of the EDZipDataset class.

            Args:
                zip (IOBase): A file-like object representing the zip file.
                con (sqlite3.Connection): A connection to the SQLite database containing the external directory.
                limit (Sequence[str]): An optional list of filenames to limit the dataset to.
        """
        self.edzip = edzip.EDZipFile(zip, con)
        if limit is not None:
            self.infolist = list(self.edzip.getinfos(limit))
        else:
            self.infolist = self.edzip.infolist()
        self.transform = transform
        
    def __len__(self):
        return len(self.infolist)
    
    def __getitem__(self, idx: int) -> T:
        return self.transform(self.edzip, idx, self.infolist[idx])

    def __getitems__(self, idxs: list[int]) -> list[T]:
        return [self.transform(self.edzip,idx,info) for idx,info in zip(idxs,self.edzip.getinfos(idxs))]


class S3HostedEDZipDataset(EDZipDataset[T]):
    """A dataset class for reading data from an S3 hosted zip file with an external sqlite3 directory."""

    def __init__(self, zip_url:str, sqlite_dir: str, s3_client = None, *args, **kwargs):
        """Creates a new instance of the S3HostedEDZipDataset class.

            Args:
                zip_url (str): The URL of the zip file on S3.
                sqlite_dir (str): The directory containing the sqlite3 database file ().
                s3_client (boto3.client): The S3 client object to use.
        """
        zf = smart_open.open(zip_url, "rb", transport_params=dict(client=s3_client))
        sqfname = zf.name+".offsets.sqlite3"
        sqfpath = f"{sqlite_dir}/{sqfname}"        
        if not os.path.exists(sqfpath):
            if s3_client is None:
                raise ValueError("s3_client must be provided if the sqlite3 file does not already exist")
            with smart_open.open(f"{zip_url}.offsets.sqlite3", "rb", transport_params=dict(client=s3_client)) as sf:
                os.makedirs(os.path.dirname(sqfpath), exist_ok=True)
                with open(sqfpath, "wb") as df:
                    shutil.copyfileobj(sf, df)
        super().__init__(zf, sqlite3.connect(sqfpath), *args, **kwargs)


T = TypeVar('T')

class LinearSubset(Dataset[T]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T]
    start: int
    end: int

    def __init__(self, dataset: Dataset[T], start: int = 0, end: Union[int,None] = None) -> None:
        self.dataset = dataset
        self.start = start
        if end is not None:
            self.end = end
        else: 
            self.end = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

    def __getitems__(self, indices: list[int]) -> list[T]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.start + idx for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.start + idx] for idx in indices]

    def __len__(self):
        return self.end - self.start


T2 = TypeVar('T2')

class TransformedDataset(Dataset[T2]):
    r"""Create a transformed dataset by applying a transform function to all samples.

    Args:
        dataset (Dataset[T]): The underlying dataset
        transform (Callable[T,T2]): The transformation function to be applied to each sample
    """
    dataset: Dataset[T]
    transform: Callable[[T2],T]

    def __init__(self, dataset: Dataset[T], transform: Callable[[T2],T]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __getitems__(self, indices: list[int]) -> list[T]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return [self.transform(item) for item in self.dataset.__getitems__(indices)]  # type: ignore[attr-defined]
        else:
            return [self.transform(self.dataset[idx]) for idx in indices]

    def __len__(self):
        return len(self.dataset)

__all__ = ["EDZipDataset","S3HostedEDZipDataset","LinearSubset","TransformedDataset","get_s3_client"]