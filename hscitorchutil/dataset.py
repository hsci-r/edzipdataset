import abc
import bisect
import random
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, TypeVar, Generic, TypeVarTuple, cast
import torch
import lightning.pytorch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torch.utils.data.dataloader
import logging

T_co = TypeVar('T_co', covariant=True)
T2_co = TypeVar('T2_co', covariant=True)
T3_co = TypeVar('T3_co', covariant=True)
Ts = TypeVarTuple("Ts")


class LinearMapSubset(Dataset[T_co], Generic[T_co]):
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
            self.end = len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.start + idx for idx in indices])  # type: ignore[attr-defined] # noqa
        else:
            return [self.dataset[self.start + idx] for idx in indices]

    def __len__(self):
        return self.end - self.start


K_co = TypeVar('K_co', covariant=True)
K2_co = TypeVar('K2_co', covariant=True)
T2_co = TypeVar('T2_co', covariant=True)


def identity_transformation(i: T_co) -> T_co:
    return i


class KeyTransformingMapDataset(Dataset[T_co], Generic[K_co, K2_co, T_co]):
    r"""Create a transformed map dataset by applying a transform function to all samples.

    Args:
        dataset (Dataset[T_co]): The underlying map dataset
        transform (Callable[T:co,T2_co]): The transformation function to be applied to each sample
    """

    def __init__(self, dataset: Dataset[T_co], transform: Callable[[Sequence[K_co]], Sequence[K2_co]]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.dataset[self.transform([idx])[0]]

    def __getitems__(self, indices: Sequence[K_co]) -> Sequence[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__(self.transform(indices))  # type: ignore[attr-defined] # noqa
        else:
            return [self.dataset[idx] for idx in self.transform(indices)]

    def __len__(self):
        return len(self.dataset)  # type: ignore[attr-defined]


class EntryTransformingMapDataset(Dataset[T2_co], Generic[T_co, T2_co]):
    r"""Create a transformed map dataset by applying a transform function to all samples.

    Args:
        dataset (Dataset[T_co]): The underlying map dataset
        transform (Callable[T:co,T2_co]): The transformation function to be applied to each sample
    """

    def __init__(self, dataset: Dataset[T_co], transform: Callable[[Sequence[T_co]], Sequence[T2_co]]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform([self.dataset[idx]])[0]

    def __getitems__(self, indices: Sequence[K_co]) -> Sequence[T2_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.transform(self.dataset.__getitems__(indices))  # type: ignore[attr-defined] # noqa
        else:
            return self.transform([self.dataset[idx] for idx in indices])

    def __len__(self):
        return len(self.dataset)  # type: ignore[attr-defined]


class ShuffledMapDataset(Dataset[T_co], Generic[T_co]):
    r"""
    Shuffle the input map dataset via its indices.

    Args:
        dataset (Dataset): Map dataset being shuffled
        seed: (int, optional): The seed to be used for shuffling. If not provided, the current time is used.
        indices (list[Any]): a list of indices for the parent Dataset. If not provided, we assume it uses 0-based indexing
    """
    dataset: Dataset[T_co]

    def __init__(self, dataset: Dataset[T_co], seed: int, indices: Optional[list[Any]] = None) -> None:
        self.dataset = dataset
        self.seed = seed
        self.indices = indices
        self._shuffle()

    def _shuffle(self):
        if self.indices is None:
            rng = torch.Generator().manual_seed(self.seed)
            self._shuffled_indices = torch.randperm(
                len(self.dataset), generator=rng).tolist()  # type: ignore
        else:
            rng = random.Random()
            rng.seed(self.seed)
            self._shuffled_indices: list = rng.sample(
                self.indices, len(self.indices))

    def __getitem__(self, idx):
        return self.dataset[self._shuffled_indices[idx]]

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self._shuffled_indices[idx] for idx in indices])  # type: ignore[attr-defined] # noqa
        else:
            return [self.dataset[self._shuffled_indices[idx]] for idx in indices]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getstate__(self):
        state = (
            self.dataset,
            self.indices,
            self.seed,
        )
        return state

    def __setstate__(self, state):
        (
            self.dataset,
            self.indices,
            self.seed,
        ) = state
        self._shuffle()


def _log_exception(ds: 'ExceptionHandlingMapDataset', idx: int, e: Exception) -> None:
    logging.exception(
        f"ExceptionHandlingMapDataset encountered exception at index {idx}. Returning None.")


class ExceptionHandlingMapDataset(Dataset[T_co], Generic[T_co]):
    r"""A dataset wrapper that catches exceptions and instead of bailing out, returns None.

    Args:
        dataset (Dataset[T_co]): The underlying map dataset
        on_exception (Callable[[int, Exception],Any]): The function to be called when an exception is raised.
    """

    def __init__(self, dataset: Dataset[T_co], on_exception: Callable[['ExceptionHandlingMapDataset', int, Exception], T_co] = _log_exception) -> None:
        self.dataset = dataset
        self.on_exception = on_exception

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception as e:
            return self.on_exception(self, idx, e)

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            try:
                return self.dataset.__getitems__(indices)  # type: ignore[attr-defined] # noqa
            except Exception:
                return [self.__getitem__(idx) for idx in indices]
        else:
            return [self.__getitem__(idx) for idx in indices]  # type: ignore

    def __len__(self):
        return len(self.dataset)  # type: ignore


class DatasetToIterableDataset(torch.utils.data.IterableDataset[T_co], Generic[T_co]):
    def __init__(self, dataset: torch.utils.data.Dataset[T_co]):
        self.dataset = dataset

    def __iter__(self):
        if hasattr(self.dataset, "__iter__"):
            return self.dataset.__iter__()  # type: ignore
        for i in range(len(self.dataset)):  # type: ignore
            yield self.dataset[i]


class UnionMapDataset(Dataset[T_co], Generic[T_co]):
    def __init__(self, datasets: Sequence[Dataset[T_co]]) -> None:
        self.datasets = datasets
        self.supports_getitems = True
        start = 0
        self.start_offsets = []
        for dataset in datasets:
            if not callable(getattr(dataset, "__getitems__", None)):
                self.supports_getitems = False
            self.start_offsets.append(start)
            start += len(dataset)  # type: ignore
        self._len = start

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.start_offsets, idx) - 1
        return self.datasets[dataset_idx][idx - self.start_offsets[dataset_idx]]

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if self.supports_getitems:
            idxs_by_datasets = [[] for _ in self.datasets]
            for idx_idx, idx in enumerate(indices):
                dataset_idx = bisect.bisect_right(self.start_offsets, idx) - 1
                idxs_by_datasets[dataset_idx].append(
                    (idx - self.start_offsets[dataset_idx], idx_idx))
            items = [None for _ in indices]
            for dataset_idx, idxs_by_dataset in enumerate(idxs_by_datasets):
                if idxs_by_dataset:
                    dataset_items = self.datasets[dataset_idx].__getitems__(  # type: ignore
                        [idx for idx, _ in idxs_by_dataset])
                    for (_, idx), item in zip(idxs_by_dataset, dataset_items):
                        items[idx] = item
            return items  # type: ignore
        else:
            return [self.__getitem__(idx) for idx in indices]

    def __len__(self):
        return self._len


class TypedDataLoader(Iterable[T_co], DataLoader[T_co], Generic[T_co]):
    pass


def remove_nones_from_batch(batch: Sequence[T_co], collate_fn: Callable[[Any], Any] = torch.utils.data.dataloader.default_collate) -> Sequence[T2_co]:  # type: ignore
    """Removes None values from batch. Used to recover from errors."""
    batch = list(filter(lambda x: x is not None, batch))
    if batch:
        try:
            return collate_fn(batch)
        except Exception as e:
            logging.exception("Failed to collate batch, returning empty batch")
            return ()
    logging.warn("Batch is empty")
    return ()


class ABaseDataModule(lightning.pytorch.LightningDataModule, Generic[T_co, T2_co], abc.ABC):
    def __init__(self,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 prepare_data_per_node: bool = True,
                 collate_fn: Optional[Callable[[Sequence[T_co]], T2_co]] = remove_nones_from_batch):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data_per_node = prepare_data_per_node
        self.collate_fn = collate_fn
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        super().__init__()

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        super().prepare_data()

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        # Called at the beginning of fit (train + validate), validate, test, or predict.
        # This is a good hook when you need to build models dynamically or adjust something
        # about them. This hook is called on every process when using DDP.
        super().setup(stage)

    def train_dataloader(self) -> TypedDataLoader[T2_co]:
        if self.train_dataset is None:
            raise ValueError("Training dataset not available")
        return cast(TypedDataLoader[T2_co], DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                                                       collate_fn=self.collate_fn, pin_memory=True))

    def val_dataloader(self) -> TypedDataLoader[T2_co]:
        if self.val_dataset is None:
            raise ValueError("Validation dataset not available")
        return cast(TypedDataLoader[T2_co], DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, collate_fn=self.collate_fn, pin_memory=True))

    def test_dataloader(self) -> TypedDataLoader[T2_co]:
        if self.test_dataset is None:
            raise ValueError("Test dataset not available")
        return cast(TypedDataLoader[T2_co], DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, collate_fn=self.collate_fn, pin_memory=True))

    def predict_dataloader(self) -> TypedDataLoader[T2_co]:
        if self.predict_dataset is None:
            raise ValueError("Predict dataset not available")
        return cast(TypedDataLoader[T2_co], DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, collate_fn=self.collate_fn, pin_memory=True))

    def teardown(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        super().teardown(stage)
