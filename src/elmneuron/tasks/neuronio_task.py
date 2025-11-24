"""
NeuronIO task for biophysical neuron modeling.

This module provides a Lightning wrapper for ELM models trained on
the NeuronIO dataset, with task-specific loss functions, metrics,
and postprocessing.
"""

from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchmetrics import MeanAbsoluteError, MeanSquaredError
    from torchmetrics.classification import BinaryAUROC

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

from ..expressive_leaky_memory_neuron_v2 import ELM
from ..neuronio.neuronio_data_utils import DEFAULT_Y_TRAIN_SOMA_SCALE


class NeuronIOTask(pl.LightningModule):
    """
    Lightning module for NeuronIO biophysical neuron modeling task.

    This task predicts:
    1. Spike probability (binary classification with sigmoid)
    2. Soma voltage (regression with scaling)

    The model uses a combination of BCE loss for spikes and MSE loss
    for soma voltage, and tracks AUC and MSE metrics.
    """

    def __init__(
        self,
        model: ELM,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: Literal["adam", "adamw", "sgd"] = "adam",
        scheduler: Literal["none", "cosine", "step"] | None = None,
        scheduler_kwargs: dict | None = None,
        spike_loss_weight: float = 0.5,
        soma_loss_weight: float = 0.5,
        y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
    ):
        """
        Initialize NeuronIO task.

        Args:
            model: Base ELM model
            learning_rate: Learning rate (default: 1e-3)
            weight_decay: Weight decay for optimizer (default: 0.0)
            optimizer: Optimizer type (default: 'adam')
            scheduler: Learning rate scheduler (default: None)
            scheduler_kwargs: Additional scheduler arguments
            spike_loss_weight: Weight for spike loss (default: 1.0)
            soma_loss_weight: Weight for soma loss (default: 1.0)
            y_train_soma_scale: Soma voltage scaling factor
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.spike_loss_weight = spike_loss_weight
        self.soma_loss_weight = soma_loss_weight
        self.y_train_soma_scale = y_train_soma_scale

        # Verify model output dimension is 2 (spike + soma)
        assert model.num_output == 2, "NeuronIOTask requires model with num_output=2"

        # Initialize metrics
        if TORCHMETRICS_AVAILABLE:
            self.train_spike_auc = BinaryAUROC()
            self.val_spike_auc = BinaryAUROC()
            self.test_spike_auc = BinaryAUROC()

            self.train_soma_mse = MeanSquaredError()
            self.train_soma_mae = MeanAbsoluteError()
            self.val_soma_mse = MeanSquaredError()
            self.val_soma_mae = MeanAbsoluteError()
            self.test_soma_mse = MeanSquaredError()
            self.test_soma_mae = MeanAbsoluteError()
        else:
            print("Warning: torchmetrics not available, metrics will not be tracked")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.

        Args:
            X: Input tensor (batch, time, num_input)

        Returns:
            Raw outputs (batch, time, 2)
        """
        return self.model(X)

    def postprocess_outputs(
        self, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply NeuronIO-specific postprocessing.

        Args:
            outputs: Raw model outputs (batch, time, 2)

        Returns:
            tuple of (spike_pred, soma_pred):
            - spike_pred: Spike probabilities (batch, time)
            - soma_pred: Scaled soma voltage (batch, time)
        """
        spike_pred = torch.sigmoid(outputs[..., 0])
        soma_pred = (1 / self.y_train_soma_scale) * outputs[..., 1]
        return spike_pred, soma_pred

    def compute_loss(
        self,
        spike_pred: torch.Tensor,
        soma_pred: torch.Tensor,
        spike_target: torch.Tensor,
        soma_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            spike_pred: Spike predictions (batch, time)
            spike_target: Spike targets (batch, time)
            soma_pred: Soma predictions (batch, time)
            soma_target: Soma targets (batch, time)

        Returns:
            tuple of (total_loss, spike_loss, soma_loss)
        """
        # Binary cross-entropy for spikes
        spike_loss = F.binary_cross_entropy_with_logits(
            spike_pred, spike_target, reduction="mean"
        )

        # MSE for soma voltage
        soma_loss = F.mse_loss(soma_pred, soma_target, reduction="mean")

        # Combined loss
        total_loss = (
            self.spike_loss_weight * spike_loss + self.soma_loss_weight * soma_loss
        )

        return total_loss, spike_loss, soma_loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        X, (spike_target, soma_target) = batch

        # Forward pass
        outputs = self(X)
        spike_pred, soma_pred = self.postprocess_outputs(outputs)

        # Compute loss
        loss, spike_loss, soma_loss = self.compute_loss(
            spike_pred, soma_pred, spike_target, soma_target
        )

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/spike_loss", spike_loss, on_step=False, on_epoch=True)
        self.log("train/soma_loss", soma_loss, on_step=False, on_epoch=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            self.train_spike_auc(spike_pred.flatten(), spike_target.flatten().int())
            self.train_soma_mse(soma_pred, soma_target)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("train/spike_auc", self.train_spike_auc, prog_bar=True)
            self.log("train/soma_rmse", np.sqrt(self.train_soma_mse))
            self.log("train/soma_mae", self.train_soma_mae)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        X, (spike_target, soma_target) = batch

        # Forward pass
        outputs = self(X)
        spike_pred, soma_pred = self.postprocess_outputs(outputs)

        # Compute loss
        loss, spike_loss, soma_loss = self.compute_loss(
            spike_pred, soma_pred, spike_target, soma_target
        )

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/spike_loss", spike_loss, on_step=False, on_epoch=True)
        self.log("val/soma_loss", soma_loss, on_step=False, on_epoch=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            self.val_spike_auc(spike_pred.flatten(), spike_target.flatten().int())
            self.val_soma_mse(soma_pred, soma_target)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("val/spike_auc", self.val_spike_auc, prog_bar=True)
            self.log("val/soma_rmse", np.sqrt(self.val_soma_mse))
            self.log("val/soma_mae", self.val_soma_mae)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step."""
        X, (spike_target, soma_target) = batch

        # Forward pass
        outputs = self(X)
        spike_pred, soma_pred = self.postprocess_outputs(outputs)

        # Compute loss
        loss, spike_loss, soma_loss = self.compute_loss(
            spike_pred, soma_pred, spike_target, soma_target
        )

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/spike_loss", spike_loss, on_step=False, on_epoch=True)
        self.log("test/soma_loss", soma_loss, on_step=False, on_epoch=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            self.test_spike_auc(spike_pred.flatten(), spike_target.flatten().int())
            self.test_soma_mse(soma_pred, soma_target)

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("test/spike_auc", self.test_spike_auc)
            self.log("test/soma_rmse", np.sqrt(self.test_soma_mse))
            self.log("test/soma_mae", self.test_soma_mae)

    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler."""
        # Create optimizer
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Optionally create scheduler
        if self.scheduler_name is None or self.scheduler_name == "none":
            return optimizer

        elif self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.scheduler_kwargs
            )
        elif self.scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **self.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def predict_step(
        self, batch: tuple | torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step.

        Args:
            batch: Input batch (X only or (X, targets))
            batch_idx: Batch index

        Returns:
            tuple of (spike_pred, soma_pred)
        """
        # Handle both (X, y) and X-only batches
        if isinstance(batch, (list, tuple)):
            X = batch[0]
        else:
            X = batch

        # Forward pass
        outputs = self(X)
        spike_pred, soma_pred = self.postprocess_outputs(outputs)

        return spike_pred, soma_pred
