from io import IOBase
import os
from typing import Union
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


class EDZipDataset(Dataset):
    """A dataset class for reading data from a zip file with an external sqlite3 directory."""

    def __init__(self, zip: IOBase, con: sqlite3.Connection):
        """Creates a new instance of the EDZipDataset class.

            Args:
                zip (IOBase): A file-like object representing the zip file.
                con (sqlite3.Connection): A connection to the SQLite database containing the external directory.
        """
        self.edzip = edzip.EDZipFile(zip, con)
        self.infolist = self.edzip.infolist()
        
    def __len__(self):
        return len(self.infolist)
    
    def __getitem__(self, idx: int) -> IOBase:
        return self.edzip.open(self.infolist[idx])

    def __getitems__(self, idxs: list[int]) -> list[IOBase]:
        return [self.edzip.open(info) for info in self.edzip.getinfos(idxs)]


class S3HostedEDZipDataset(EDZipDataset):
    """A dataset class for reading data from an S3 hosted zip file with an external sqlite3 directory."""

    def __init__(self, zip_url:str, sqlite_dir: str, s3_client = None):
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
        super().__init__(zf, sqlite3.connect(sqfpath))


__all__ = ["EDZipDataset","S3HostedEDZipDataset","get_s3_client"]