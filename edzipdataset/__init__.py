from io import IOBase
import os
import random
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union
from zipfile import ZipInfo
import edzip
import smart_open
import torch
import yaml
import boto3
import sqlite3
from torch.utils.data import Dataset
import shutil
import yaml
import boto3
import dill

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


T_co = TypeVar('T_co', covariant=True)

class EDZipMapDataset(Dataset[T_co]):
    """A map dataset class for reading data from a zip file with an external sqlite3 directory."""

    def __init__(self, zip: Callable[[],IOBase], con: Callable[[],sqlite3.Connection], transform: Callable[[edzip.EDZipFile,int,ZipInfo], T_co] = lambda edzip,idx,zinfo: edzip.open(zinfo), limit: Optional[Sequence[str]] = None):
        """Creates a new instance of the EDZipDataset class.

            Args:
                zip (IOBase): A file-like object representing the zip file.
                con (sqlite3.Connection): A connection to the SQLite database containing the external directory.
                limit (Sequence[str]): An optional list of filenames to limit the dataset to.
        """
        self.zip = zip
        self.con = con
        self.transform = transform
        self.limit = limit
        self._edzip = None
        self._infolist = None

    @property
    def edzip(self) -> edzip.EDZipFile:
        if self._edzip is None:
            self._edzip = edzip.EDZipFile(self.zip(), self.con())
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
        return self.transform(self.edzip, idx, self.infolist[idx])

    def __getitems__(self, idxs: list[int]) -> list[T_co]:
        return [self.transform(self.edzip,idx,info) for idx,info in zip(idxs,self.edzip.getinfos(idxs))]
    
    def __setstate__(self, state):
        (
            self.zip, 
            self.con, 
            self.transform, 
            self.limit) = dill.loads(state)
        self._edzip = None
        self._infolist = None
    
    def __getstate__(self) -> object:
        return dill.dumps((
            self.zip, 
            self.con, 
            self.transform, 
            self.limit
        ))
    

class S3HostedEDZipMapDataset(EDZipMapDataset[T_co]):
    """A map dataset class for reading data from an S3 hosted zip file with an external sqlite3 directory."""

    def __init__(self, zip_url:str, sqlite_dir: str, s3_credentials_yaml_file: Optional[Union[str,os.PathLike]] = None, *args, **kwargs):
        """Creates a new instance of the S3HostedEDZipDataset class.

            Args:
                zip_url (str): The URL of the zip file on S3.
                sqlite_dir (str): The directory containing the sqlite3 database file ().
                s3_client (boto3.client): The S3 client object to use.
        """
        def s3_client():
            return get_s3_client(s3_credentials_yaml_file) if s3_credentials_yaml_file is not None else None
        def zip():
            return smart_open.open(zip_url, "rb", transport_params=dict(client=s3_client()))
        sqfname = os.path.basename(zip_url)+".offsets.sqlite3"
        sqfpath = f"{sqlite_dir}/{sqfname}"        
        if not os.path.exists(sqfpath):
            if s3_credentials_yaml_file is None:
                raise ValueError("s3_credentials_yaml_file must be provided if the sqlite3 file does not already exist")
            with smart_open.open(f"{zip_url}.offsets.sqlite3", "rb", transport_params=dict(client=s3_client())) as sf:
                os.makedirs(os.path.dirname(sqfpath), exist_ok=True)
                with open(sqfpath, "wb") as df:
                    shutil.copyfileobj(sf, df)
        def sqlite():
            return sqlite3.connect(sqfpath)
        super().__init__(zip=zip,con=sqlite, *args, **kwargs)


class LinearMapSubset(Dataset[T_co]):
    r"""
    Slice a map dataset at specified indices.

    Args:
        dataset (Dataset[T_co]): The whole map dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    start: int
    end: int

    def __init__(self, dataset: Dataset[T_co], start: int = 0, end: Optional[int] = None) -> None:
        self.dataset = dataset
        self.start = start
        if end is not None:
            self.end = end
        else: 
            self.end = len(self.dataset) # type: ignore

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.start + idx for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.start + idx] for idx in indices]

    def __len__(self):
        return self.end - self.start


T2_co = TypeVar('T2_co', covariant=True)

class TransformedMapDataset(Dataset[T2_co]):
    r"""Create a transformed map dataset by applying a transform function to all samples.

    Args:
        dataset (Dataset[T_co]): The underlying map dataset
        transform (Callable[T:co,T2_co]): The transformation function to be applied to each sample
    """

    def __init__(self, dataset: Dataset[T_co], transform: Callable[[T_co],T2_co]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __getitems__(self, indices: list[int]) -> list[T2_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return [self.transform(item) for item in self.dataset.__getitems__(indices)]  # type: ignore[attr-defined]
        else:
            return [self.transform(self.dataset[idx]) for idx in indices]

    def __len__(self):
        return len(self.dataset) # type: ignore
    
    def __setstate__(self, state):
        (
            self.dataset, 
            self.transform,
        ) = dill.loads(state)
    
    def __getstate__(self) -> object:
        return dill.dumps((
            self.dataset,
            self.transform,
        ))
    


class ShuffledMapDataset(Dataset[T_co]):
    r"""
    Shuffle the input map dataset via its indices.

    Args:
        dataset (Dataset): Map dataset being shuffled
        seed: (int, optional): The seed to be used for shuffling. If not provided, the current time is used.
        indices (list[Any]): a list of indices for the parent Dataset. If not provided, we assume it uses 0-based indexing
    """
    dataset: Dataset[T_co]

    def __init__(self, dataset: Dataset[T_co], seed: Optional[int] = None, indices: Optional[list[Any]] = None) -> None:
        self.dataset = dataset
        rng = random.Random()
        rng.seed(seed)
        if indices is None:
            indices = list(range(len(dataset))) # type: ignore
        self._shuffled_indices: list = rng.sample(indices, len(indices))

    def __getitem__(self, idx):
        return self.dataset[self._shuffled_indices[idx]]

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self._shuffled_indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self._shuffled_indices[idx]] for idx in indices]
        
    def __len__(self) -> int:
        return len(self.dataset) # type: ignore
    
    def __getstate__(self):
        state = (
            self.dataset,
            self._shuffled_indices,
        )
        return state

    def __setstate__(self, state):
        (
            self.dataset,
            self._shuffled_indices,
        ) = state
    

__all__ = ["EDZipMapDataset","S3HostedEDZipMapDataset","LinearMapSubset","TransformedMapDataset","ShuffledMapDataset","get_s3_client"]