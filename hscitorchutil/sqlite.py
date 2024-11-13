import os
import sqlite3
from typing import Callable, Generic, Iterable, Literal, Optional, TypeVar, Sequence
from typing_extensions import TypeVarTuple, Unpack
from zipfile import ZipFile
import click
import fsspec
from contextlib import closing
from torch.utils.data import Dataset, DataLoader
from edzip.sqlite import create_sqlite_directory_from_zip
from hscitorchutil.dataset import ABaseDataModule, identity_transformation
from hscifsspecutil import get_s3fs_credentials, cache_locally_if_remote

Ts = TypeVarTuple("Ts")
T_co = TypeVar('T_co', covariant=True)
T2_co = TypeVar('T2_co', covariant=True)


class SQLiteDataset(Dataset[T_co], Generic[Unpack[Ts], T_co]):
    def __init__(self, sqlite_filename: str, table_name: str, index_column: str, columns_to_return: str, id_column: str):
        self.sqlite_filename = sqlite_filename
        self.table_name = table_name
        self.index_column = index_column
        self.id_column = id_column
        self.columns_to_return = columns_to_return
        self.sqlite = sqlite3.connect(sqlite_filename)
        self._len = None

    def __len__(self):
        if self._len is None:
            with closing(self.sqlite.execute(
                    f"SELECT COUNT(*) FROM {self.table_name}")) as cur:
                self._len = cur.fetchall()[0][0]
        return self._len

    def __getitem__(self, idx: int | str) -> T_co:
        return self.__getitems__([idx])[0]

    def __getitems__(self, idxs: Sequence[int | str]) -> Sequence[T_co]:
        if isinstance(idxs[0], int):
            return self.sqlite.execute(f"SELECT {self.columns_to_return} FROM {self.table_name} WHERE {self.index_column} IN (%s)" % ','.join(
                '?' * len(idxs)), idxs).fetchall()
        else:
            return self.sqlite.execute(f"SELECT {self.columns_to_return} FROM {self.table_name} WHERE {self.id_column} IN (%s)" % ','.join(
                '?' * len(idxs)), idxs).fetchall()

    def __setstate__(self, state):
        (
            self.sqlite_filename,
            self.table_name,
            self.index_column,
            self.columns_to_return,
            self.id_column,
            self._len
        ) = state
        self.sqlite = sqlite3.connect(self.sqlite_filename)

    def __getstate__(self) -> object:
        return (
            self.sqlite_filename,
            self.table_name,
            self.index_column,
            self.columns_to_return,
            self.id_column,
            self._len
        )


class TypedDataLoader(Iterable[T_co], DataLoader[T_co], Generic[T_co]):
    pass


class SQLiteDataModule(ABaseDataModule[T_co, T2_co], Generic[Unpack[Ts], T_co, T2_co]):
    def __init__(self,
                 train_sqlite_url: str,
                 val_sqlite_url: str,
                 test_sqlite_url: str,
                 cache_dir: str,
                 table_name: str,
                 index_column: str,
                 columns_to_return: str,
                 id_column: str,
                 storage_options: dict = dict(),
                 train_transform: Callable[[
                     Dataset[tuple[Unpack[Ts]]]], Dataset[T_co]] = identity_transformation,
                 test_transform: Callable[[
                     Dataset[tuple[Unpack[Ts]]]], Dataset[T_co]] = identity_transformation,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_sqlite_url = train_sqlite_url
        self.val_sqlite_url = val_sqlite_url
        self.test_sqlite_url = test_sqlite_url
        self.cache_dir = cache_dir
        self.storage_options = storage_options

        self.table_name = table_name
        self.index_column = index_column
        self.columns_to_return = columns_to_return
        self.id_column = id_column
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_sqlite_databases(self):
        """Ensure sqlite databases are downloaded"""
        cache_locally_if_remote(
            self.train_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir)
        cache_locally_if_remote(
            self.val_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir)
        cache_locally_if_remote(
            self.test_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir)

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        """Fulfil the requirements for a given stage"""
        if (stage == "fit") and self.train_dataset is None:
            self.train_dataset = self.train_transform(SQLiteDataset(cache_locally_if_remote(
                self.train_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir),
                self.table_name,
                self.index_column,
                self.columns_to_return,
                self.id_column))
        if (stage == "fit" or stage == "validate") and self.val_dataset is None:
            self.val_dataset = self.test_transform(SQLiteDataset(cache_locally_if_remote(
                self.val_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir),
                self.table_name,
                self.index_column,
                self.columns_to_return,
                self.id_column))
        if (stage == "test") and self.test_dataset is None:
            self.test_dataset = self.test_transform(SQLiteDataset(cache_locally_if_remote(
                self.test_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir),
                self.table_name,
                self.index_column,
                self.columns_to_return,
                self.id_column))


@click.command()
@click.option("--key", required=False)
@click.option("--secret", required=False, help="AWS secret access key or file from which to read credentials")
@click.option("--endpoint-url", required=False)
@click.argument("zip-url")
@click.argument("sqlite-filename", required=True)
def main(zip_url: str, sqlite_filename: Optional[str] = None, endpoint_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None):
    if secret is not None and os.path.exists(secret):
        credentials = get_s3fs_credentials(secret)
    else:
        credentials = dict()
    with fsspec.open(zip_url, **credentials) as zf:  # type: ignore
        create_sqlite_directory_from_zip(
            ZipFile(zf), sqlite_filename)  # type: ignore


if __name__ == "__main__":
    main()
