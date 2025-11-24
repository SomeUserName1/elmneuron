"""
Classification task for sequence classification.

This module provides a Lightning wrapper for ELM models on
classification tasks (MNIST, CIFAR, etc.) with various temporal
pooling strategies.
"""

from typing import Any, Literal

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchmetrics import Accuracy, ConfusionMatrix

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

from ..expressive_leaky_memory_neuron_v2 import ELM


class ClassificationTask(pl.LightningModule):
    """
    Lightning module for sequence classification tasks.

    Supports various temporal pooling strategies to convert
    sequence outputs to class predictions.
    """

    def __init__(
        self,
        model: ELM,
        num_classes: int | None = None,
        pooling: Literal["last", "mean", "max", "attention"] = "last",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: Literal["adam", "adamw", "sgd"] = "adam",
        scheduler: Literal["none", "cosine", "step"] | None = None,
        scheduler_kwargs: dict | None = None,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize classification task.

        Args:
            model: Base ELM model
            num_classes: Number of classes (inferred from model if None)
            pooling: Temporal pooling strategy:
                - 'last': Use final timestep output
                - 'mean': Average over time
                - 'max': Max over time
                - 'attention': Learned attention pooling
            learning_rate: Learning rate (default: 1e-3)
            weight_decay: Weight decay (default: 0.0)
            optimizer: Optimizer type (default: 'adam')
            scheduler: Learning rate scheduler (default: None)
            scheduler_kwargs: Additional scheduler arguments
            label_smoothing: Label smoothing factor (default: 0.0)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes if num_classes is not None else model.num_output
        self.pooling = pooling
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.label_smoothing = label_smoothing

        # Verify model output dimension matches num_classes
        assert (
            model.num_output == self.num_classes
        ), f"Model output dimension ({model.num_output}) must match num_classes ({self.num_classes})"

        # Initialize attention pooling if needed
        if pooling == "attention":
            self.attention = nn.Linear(model.num_output, 1)
        else:
            self.attention = None

        # Initialize metrics
        if TORCHMETRICS_AVAILABLE:
            task_type = "multiclass" if self.num_classes > 2 else "binary"

            self.train_acc = Accuracy(task=task_type, num_classes=self.num_classes)
            self.val_acc = Accuracy(task=task_type, num_classes=self.num_classes)
            self.test_acc = Accuracy(task=task_type, num_classes=self.num_classes)

            # Top-5 accuracy for large number of classes
            if self.num_classes > 5:
                self.val_acc_top5 = Accuracy(
                    task=task_type, num_classes=self.num_classes, top_k=5
                )
                self.test_acc_top5 = Accuracy(
                    task=task_type, num_classes=self.num_classes, top_k=5
                )
        else:
            print("Warning: torchmetrics not available, metrics will not be tracked")

    def pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence to single representation.

        Args:
            x: Sequence tensor (batch, time, features)

        Returns:
            Pooled tensor (batch, features)
        """
        if self.pooling == "last":
            # Use final timestep
            return x[:, -1]

        elif self.pooling == "mean":
            # Average over time
            return x.mean(dim=1)

        elif self.pooling == "max":
            # Max over time
            return x.max(dim=1)[0]

        elif self.pooling == "attention":
            # Learned attention pooling
            # Compute attention scores
            attn_scores = self.attention(x)  # (batch, time, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, time, 1)

            # Weighted sum
            pooled = (x * attn_weights).sum(dim=1)  # (batch, features)
            return pooled

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model with pooling.

        Args:
            X: Input tensor (batch, time, num_input)

        Returns:
            Class logits (batch, num_classes)
        """
        # Get sequence outputs
        outputs = self.model(X)  # (batch, time, num_classes)

        # Pool to class logits
        logits = self.pool_sequence(outputs)  # (batch, num_classes)

        return logits

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        X, targets = batch

        # Forward pass
        logits = self(X)

        # Compute loss
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            preds = logits.argmax(dim=-1)
            self.train_acc(preds, targets)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("train/acc", self.train_acc, prog_bar=True)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        X, targets = batch

        # Forward pass
        logits = self(X)

        # Compute loss
        loss = F.cross_entropy(logits, targets)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            preds = logits.argmax(dim=-1)
            self.val_acc(preds, targets)

            if self.num_classes > 5:
                self.val_acc_top5(logits, targets)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("val/acc", self.val_acc, prog_bar=True)

            if self.num_classes > 5:
                self.log("val/acc_top5", self.val_acc_top5)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step."""
        X, targets = batch

        # Forward pass
        logits = self(X)

        # Compute loss
        loss = F.cross_entropy(logits, targets)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        # Update metrics
        if TORCHMETRICS_AVAILABLE:
            preds = logits.argmax(dim=-1)
            self.test_acc(preds, targets)

            if self.num_classes > 5:
                self.test_acc_top5(logits, targets)

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        if TORCHMETRICS_AVAILABLE:
            self.log("test/acc", self.test_acc)

            if self.num_classes > 5:
                self.log("test/acc_top5", self.test_acc_top5)

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
            Class predictions (batch,)
        """
        # Handle both (X, y) and X-only batches
        if isinstance(batch, (list, tuple)):
            X = batch[0]
        else:
            X = batch

        # Forward pass
        logits = self(X)
        preds = logits.argmax(dim=-1)

        return preds
