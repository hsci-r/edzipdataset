import bisect
import random
from typing import Any, Callable, Optional, Sequence, TypeVar
import torch
from torch.utils.data import Dataset
import logging

T_co = TypeVar('T_co', covariant=True)


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


T2_co = TypeVar('T2_co', covariant=True)


class TransformedMapDataset(Dataset[T2_co]):
    r"""Create a transformed map dataset by applying a transform function to all samples.

    Args:
        dataset (Dataset[T_co]): The underlying map dataset
        transform (Callable[T:co,T2_co]): The transformation function to be applied to each sample
    """

    def __init__(self, dataset: Dataset[T_co], transform: Callable[..., T2_co]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __getitems__(self, indices: list[int]) -> list[T2_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return [self.transform(item) for item in self.dataset.__getitems__(indices)]  # type: ignore[attr-defined] # noqa
        else:
            return [self.transform(self.dataset[idx]) for idx in indices]

    def __len__(self):
        return len(self.dataset)  # type: ignore


class ShuffledMapDataset(Dataset[T_co]):
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


class ExceptionHandlingMapDataset(Dataset[T_co]):
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


class DatasetToIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __iter__(self):
        if hasattr(self.dataset, "__iter__"):
            return self.dataset.__iter__()
        for i in range(len(self.dataset)):
            yield self.dataset[i]


class UnionMapDataset(Dataset[T_co]):
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


__all__ = ["ExceptionHandlingMapDataset", "LinearMapSubset", "TransformedMapDataset",
           "ShuffledMapDataset", "UnionMapDataset", "DatasetToIterableDataset"]
