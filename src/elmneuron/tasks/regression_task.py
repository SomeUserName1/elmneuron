"""
Regression task for sequence regression.

This module provides a Lightning wrapper for ELM models on
regression tasks with various temporal pooling strategies.
"""

from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

from ..expressive_leaky_memory_neuron_v2 import ELM


class RegressionTask(pl.LightningModule):
    """
    Lightning module for sequence regression tasks.

    Supports various temporal pooling strategies and loss functions.
    """

    def __init__(
        self,
        model: ELM,
        pooling: Literal["last", "mean", "max", "none"] = "last",
        loss_fn: Literal["mse", "mae", "huber"] = "mse",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: Literal["adam", "adamw", "sgd"] = "adam",
        scheduler: Literal["none", "cosine", "step"] | None = None,
        scheduler_kwargs: dict | None = None,
    ):
        """
        Initialize regression task.

        Args:
            model: Base ELM model
            pooling: Temporal pooling strategy:
                - 'last': Use final timestep output
                - 'mean': Average over time
                - 'max': Max over time
                - 'none': No pooling (sequence-to-sequence)
            loss_fn: Loss function ('mse', 'mae', or 'huber')
            learning_rate: Learning rate (default: 1e-3)
            weight_decay: Weight decay (default: 0.0)
            optimizer: Optimizer type (default: 'adam')
            scheduler: Learning rate scheduler (default: None)
            scheduler_kwargs: Additional scheduler arguments
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.pooling = pooling
        self.loss_fn_name = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Select loss function
        if loss_fn == "mse":
            self.loss_fn = F.mse_loss
        elif loss_fn == "mae":
            self.loss_fn = F.l1_loss
        elif loss_fn == "huber":
            self.loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Initialize metrics
        if TORCHMETRICS_AVAILABLE:
            self.train_mse = MeanSquaredError()
            self.val_mse = MeanSquaredError()
            self.test_mse = MeanSquaredError()

            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            self.test_mae = MeanAbsoluteError()

            self.val_r2 = R2Score()
            self.test_r2 = R2Score()
        else:
            print("Warning: torchmetrics not available, metrics will not be tracked")

    def pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence to single representation (if pooling is not 'none').

        Args:
            x: Sequence tensor (batch, time, features)

        Returns:
            Pooled tensor (batch, features) or unchanged (batch, time, features)
        """
        if self.pooling == "none":
            # No pooling (sequence-to-sequence)
            return x

        elif self.pooling == "last":
            # Use final timestep
            return x[:, -1]

        elif self.pooling == "mean":
            # Average over time
            return x.mean(dim=1)

        elif self.pooling == "max":
            # Max over time
            return x.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model with optional pooling.

        Args:
            X: Input tensor (batch, time, num_input)

        Returns:
            Predictions (batch, num_output) or (batch, time, num_output)
        """
        # Get sequence outputs
        outputs = self.model(X)  # (batch, time, num_output)

        # Apply pooling
        predictions = self.pool_sequence(outputs)

        return predictions

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        X, targets = batch

        # Forward pass
        predictions = self(X)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            self.train_mse(predictions, targets)
            self.train_mae(predictions, targets)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("train/mse", self.train_mse)
            self.log("train/mae", self.train_mae, prog_bar=True)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        X, targets = batch

        # Forward pass
        predictions = self(X)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            self.val_mse(predictions, targets)
            self.val_mae(predictions, targets)
            self.val_r2(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("val/mse", self.val_mse)
            self.log("val/mae", self.val_mae, prog_bar=True)
            self.log("val/r2", self.val_r2)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step."""
        X, targets = batch

        # Forward pass
        predictions = self(X)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            self.test_mse(predictions, targets)
            self.test_mae(predictions, targets)
            self.test_r2(predictions, targets)

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("test/mse", self.test_mse)
            self.log("test/mae", self.test_mae)
            self.log("test/r2", self.test_r2)

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

    def predict_step(self, batch: tuple | torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Prediction step.

        Args:
            batch: Input batch (X only or (X, targets))
            batch_idx: Batch index

        Returns:
            Predictions
        """
        # Handle both (X, y) and X-only batches
        if isinstance(batch, (list, tuple)):
            X = batch[0]
        else:
            X = batch

        # Forward pass
        predictions = self(X)

        return predictions
