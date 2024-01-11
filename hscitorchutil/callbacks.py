import os
from lightning import LightningModule, Trainer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torch


class CUDAMemorySnapshotCallback(Callback):

    def __init__(self, save_path: str | None = None) -> None:
        self.save_path = save_path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        torch.cuda.memory._record_memory_history()

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self.save_path is None:
            if len(trainer.loggers) > 0 and trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
                name = trainer.loggers[0].name
                version = trainer.loggers[0].version
                version = version if isinstance(
                    version, str) else f"version_{version}"
                self.save_path = os.path.join(
                    save_dir, str(name), version, f"memory_snapshot_rank_{trainer.global_rank}.pt")
            else:
                self.save_path = os.path.join(
                    trainer.default_root_dir, f"memory_snapshot_rank_{trainer.global_rank}.pt")
        torch.cuda.memory._dump_snapshot(self.save_path)
