import os
from typing import Callable, Generic, Literal, Optional, Sequence, TypeVar
from lightning import LightningModule, Trainer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, BasePredictionWriter
import torch

from hscitorchutil.dataset import identity_transformation


def _get_save_path(trainer: Trainer) -> str:
    if len(trainer.loggers) > 0 and trainer.loggers[0].save_dir is not None:
        save_dir = trainer.loggers[0].save_dir
        name = trainer.loggers[0].name
        version = trainer.loggers[0].version
        version = version if isinstance(
            version, str) else f"version_{version}"
        return os.path.join(
            save_dir, str(name), version)
    else:
        return trainer.default_root_dir


class CUDAMemorySnapshotCallback(Callback):

    def __init__(self, save_path: str | None = None) -> None:
        self.save_path = save_path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        torch.cuda.memory._record_memory_history()

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self.save_path is None:
            self.save_path = os.path.join(_get_save_path(
                trainer), f"memory_snapshot_rank_{trainer.global_rank}.pt")
        torch.cuda.memory._dump_snapshot(self.save_path)


T = TypeVar('T')
T2 = TypeVar('T2')


class PredictionWriterCallback(BasePredictionWriter, Generic[T, T2]):
    def __init__(self, write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch", save_path: str | None = None, transform_batch: Callable[[T], T] = identity_transformation, transform_outputs: Callable[[T2], T2] = identity_transformation) -> None:
        super().__init__(write_interval=write_interval)
        self.save_path = save_path
        self.transform_batch = transform_batch
        self.transform_outputs = transform_outputs

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self.save_path is None:
            self.save_path = os.path.join(_get_save_path(
                trainer), f"outputs_rank_{trainer.global_rank}")
            os.makedirs(self.save_path, exist_ok=True)

    def write_on_batch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           prediction: T2,
                           batch_indices: Optional[Sequence[int]],
                           batch: T,
                           batch_idx: int,
                           dataloader_idx: int,) -> None:
        torch.save(self.transform_outputs(prediction), os.path.join(self.save_path,  # type: ignore
                   f"outputs_{dataloader_idx}_{batch_idx}.pt"))
        torch.save(self.transform_batch(batch), os.path.join(self.save_path,  # type: ignore
                   f"batch_{dataloader_idx}_{batch_idx}.pt"))
