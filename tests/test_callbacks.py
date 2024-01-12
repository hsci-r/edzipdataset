
import pytest
import os

from hscitorchutil.callbacks import PredictionWriterCallback

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
import torch
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_predictionwritercallback(tmp_path: os.PathLike):
    dm = BoringDataModule()
    model = BoringModel()
    trainer = Trainer(
        max_epochs=1,
        callbacks=[PredictionWriterCallback(save_path=str(
            tmp_path), batch_transform=lambda x: x[:, 0])]
    )
    trainer.predict(model, dm)
    assert os.path.exists(os.path.join(tmp_path, "outputs_0_0.pt"))
    assert os.path.exists(os.path.join(tmp_path, "batch_0_0.pt"))
    assert torch.load(os.path.join(tmp_path, "outputs_0_0.pt")).shape == (1, 2)
    batch = torch.load(os.path.join(
        tmp_path, "batch_0_0.pt"))
    assert batch.shape == (1,)
    assert torch.equal(batch.squeeze(0).cpu(),
                       dm.predict_dataloader().dataset[0][0])
