from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torchmetrics import Accuracy


class GNN_Abstract_Base_Class(pl.LightningModule):
    def __init__(self, model, optim_args):
        super().__init__()

        self.model = model
        self.initial_lr = optim_args["lr"]
        self.scheduler_args = {
            "patience": optim_args["scheduler_patience"],
            "factor": optim_args["scheduler_factor"],
        }
        self.scheduler_monitor = optim_args["scheduler_monitor"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, **(self.scheduler_args)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.scheduler_monitor,
        }

    def _log_lr(self) -> None:
        optim = self.optimizers()
        if optim:
            for param_group in optim.param_groups:
                lr = float(param_group["lr"])
                break
            self.log("lr", lr)

    def on_train_epoch_start(self) -> None:
        self._log_lr()


class RSFGNN(GNN_Abstract_Base_Class):
    def __init__(self, model, optim_args):
        super().__init__(model, optim_args)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        g, target = batch
        x = self(g.unsqueeze(0)).squeeze(0)
        loss = self.loss(x, target)
        self.log("train_loss", loss, batch_size=1, on_epoch=True)
        accuracy = Accuracy("binary")
        acc = accuracy(x, target)
        self.log("train_acc", acc, batch_size=1, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        g, target = batch
        x = self(g.unsqueeze(0)).squeeze(0)
        loss = self.loss(x, target)
        self.log("val_loss", loss, batch_size=1, on_epoch=True)
        accuracy = Accuracy("binary")
        acc = accuracy(x, target)
        self.log("val_acc", acc, batch_size=1, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        g, target = batch
        x = self(g.unsqueeze(0)).squeeze(0)
        loss = self.loss(x, target)
        self.log("test_loss", loss, batch_size=1, on_epoch=True)
        accuracy = Accuracy("binary")
        acc = accuracy(x, target)
        self.log("test_acc", acc, batch_size=1, on_epoch=True)
        return loss
