"""Abstract base class for all forecast models."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseForecaster(pl.LightningModule, ABC):
    """Shared interface for all deep learning forecasting models."""

    def __init__(self, input_size: int, output_size: int = 1,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.input_size   = input_size
        self.output_size  = output_size
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, seq_len, input_size)"""
        ...

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss  = nn.MSELoss()(y_hat.squeeze(), y.squeeze())
        return loss, y_hat.squeeze(), y.squeeze()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch)
        self.log("test_loss", loss)
        return {"preds": y_hat, "targets": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
