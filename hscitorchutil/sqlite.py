import os
import sqlite3
from typing import Callable, Generic, Optional, Tuple, TypeVar, Sequence
from zipfile import ZipFile
import click
import fsspec
from torch.utils.data import Dataset, DataLoader
import torch
import lightning.pytorch
from edzip.sqlite import create_sqlite_directory_from_zip
from hscitorchutil.fsspec import get_s3fs_credentials, cache_locally_if_remote

T_co = TypeVar('T_co', covariant=True)
T2_co = TypeVar('T2_co', covariant=True)


def _identity(x: Sequence[Tuple]) -> Sequence[Tuple]:
    return x


class SQLiteDataset(Dataset[T2_co], Generic[T_co, T2_co]):
    # type: ignore
    def __init__(self, sqlite_filename: str, table_name: str, index_column: str, columns_to_return: str, transform: Callable[[Sequence[T_co]], Sequence[T2_co]] = _identity):
        self.sqlite_filename = sqlite_filename
        self.table_name = table_name
        self.index_column = index_column
        self.columns_to_return = columns_to_return
        self.transform = transform
        self.sqlite = sqlite3.connect(sqlite_filename)
        self._len = self.sqlite.execute(
            f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> T2_co:
        return self.__getitems__([idx])[0]

    def __getitems__(self, idxs: Sequence[int]) -> Sequence[T2_co]:
        return self.transform(
            self.sqlite.execute(f"SELECT {self.columns_to_return} FROM {self.table_name} WHERE {self.index_column} IN (%s)" % ','.join(
                '?' * len(idxs)), idxs).fetchall(),
        )

    def __setstate__(self, state):
        (
            self.sqlite_filename,
            self.table_name,
            self.index_column,
            self.columns_to_return,
            self.transform,
            self._len
        ) = state
        self.sqlite = sqlite3.connect(self.sqlite_filename)

    def __getstate__(self) -> object:
        return (
            self.sqlite_filename,
            self.table_name,
            self.index_column,
            self.columns_to_return,
            self.transform,
            self._len
        )


def _remove_nones_from_batch(batch):
    """Removes None values from batch. Used to recover from errors."""
    batch = list(filter(lambda x: x is not None, batch))
    if batch:
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except TypeError:
            return None
    return None


class SQLiteDataModule(lightning.pytorch.LightningDataModule, Generic[T_co, T2_co]):
    def __init__(self,
                 train_sqlite_url: str,
                 val_sqlite_url: str,
                 test_sqlite_url: str,
                 cache_dir: str,
                 table_name: str, 
                 index_column: str,
                 columns_to_return: str,
                 storage_options: dict = dict(),
                 batch_size: int = 64,
                 num_workers: int = 0,
                 train_transform: Callable[[Sequence[T_co]], Sequence[T2_co]] = _identity,
                 test_transform: Callable[[Sequence[T_co]], Sequence[T2_co]] = _identity,
                 prepare_data_per_node: bool = True,
                 collate_fn: Optional[Callable] = _remove_nones_from_batch):
        self.train_sqlite_url = train_sqlite_url
        self.val_sqlite_url = val_sqlite_url
        self.test_sqlite_url = test_sqlite_url
        self.cache_dir = cache_dir
        self.storage_options = storage_options
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.table_name = table_name
        self.index_column = index_column
        self.columns_to_return = columns_to_return

        self.train_transform = train_transform
        self.test_transform = test_transform
        self.prepare_data_per_node = prepare_data_per_node
        self.collate_fn = collate_fn
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        super().__init__()

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed

        # Ensure sqlite databases are downloaded
        cache_locally_if_remote(
            self.train_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir)
        cache_locally_if_remote(
            self.val_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir)
        cache_locally_if_remote(
            self.test_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir)

    def setup(self, stage: Optional[str] = None):
        if (stage is None or stage == "fit") and self.train_dataset is None:
            self.train_dataset = SQLiteDataset(cache_locally_if_remote(
                self.train_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir),
                self.table_name, 
                self.index_column,
                self.columns_to_return,
                self.train_transform)
        if (stage is None or stage == "fit" or stage == "validate") and self.val_dataset is None:
            self.val_dataset = SQLiteDataset(cache_locally_if_remote(
                self.val_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir),
                self.table_name, 
                self.index_column,
                self.columns_to_return,
                self.test_transform)
        if (stage is None or stage == "test") and self.test_dataset is None:
            self.test_dataset = SQLiteDataset(cache_locally_if_remote(
                self.test_sqlite_url, storage_options=self.storage_options, cache_dir=self.cache_dir),
                self.table_name, 
                self.index_column,
                self.columns_to_return,
                self.test_transform)

    def train_dataloader(self) -> DataLoader[T2_co]:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0,  # type: ignore
                          collate_fn=self.collate_fn, pin_memory=True)

    def val_dataloader(self) -> DataLoader[T2_co] | None:
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, collate_fn=self.collate_fn, pin_memory=True)

    def test_dataloader(self) -> DataLoader[T2_co] | None:
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, collate_fn=self.collate_fn, pin_memory=True)

    def teardown(self, stage: str):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        pass


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
