"""
Visualization callbacks for ELM models.

Provides PyTorch Lightning callbacks for monitoring training,
recording internal states, and visualizing model behavior.
"""

from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.callbacks import Callback


class StateRecorderCallback(Callback):
    """
    Records internal states of ELM model during training/validation.

    Captures branch/synapse activations and memory states for analysis
    and visualization. Useful for debugging and understanding model dynamics.

    Example:
        callback = StateRecorderCallback(
            record_every_n_epochs=5,
            num_samples=8,
        )
        trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        record_every_n_epochs: int = 5,
        num_samples: int = 8,
        record_train: bool = True,
        record_val: bool = True,
        save_dir: str | None = None,
    ):
        """
        Initialize state recorder callback.

        Args:
            record_every_n_epochs: Record states every N epochs
            num_samples: Number of samples to record per epoch
            record_train: Record during training
            record_val: Record during validation
            save_dir: Directory to save recorded states (default: None, don't save)
        """
        super().__init__()
        self.record_every_n_epochs = record_every_n_epochs
        self.num_samples = num_samples
        self.record_train = record_train
        self.record_val = record_val
        self.save_dir = Path(save_dir) if save_dir else None

        # Storage for recorded states
        self.recorded_states = {
            "train": [],
            "val": [],
        }

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record states during training."""
        if not self.record_train:
            return

        # Only record at specified epochs
        if trainer.current_epoch % self.record_every_n_epochs != 0:
            return

        # Only record first N batches
        if batch_idx >= self.num_samples:
            return

        # Get base ELM model
        base_model = self._get_base_model(pl_module)
        if base_model is None or not hasattr(base_model, "forward_with_states"):
            return

        # Record states
        with torch.no_grad():
            inputs, _ = batch
            _, states, memory = base_model.forward_with_states(inputs)

            self.recorded_states["train"].append(
                {
                    "epoch": trainer.current_epoch,
                    "batch_idx": batch_idx,
                    "states": states.cpu().numpy(),
                    "memory": memory.cpu().numpy(),
                }
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record states during validation."""
        if not self.record_val:
            return

        if trainer.current_epoch % self.record_every_n_epochs != 0:
            return

        if batch_idx >= self.num_samples:
            return

        base_model = self._get_base_model(pl_module)
        if base_model is None or not hasattr(base_model, "forward_with_states"):
            return

        with torch.no_grad():
            inputs, _ = batch
            _, states, memory = base_model.forward_with_states(inputs)

            self.recorded_states["val"].append(
                {
                    "epoch": trainer.current_epoch,
                    "batch_idx": batch_idx,
                    "states": states.cpu().numpy(),
                    "memory": memory.cpu().numpy(),
                }
            )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Save recorded states at end of epoch."""
        if not self.save_dir:
            return

        if trainer.current_epoch % self.record_every_n_epochs != 0:
            return

        # Save to disk
        epoch_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
        epoch_dir.mkdir(exist_ok=True)

        for split in ["train", "val"]:
            if self.recorded_states[split]:
                save_path = epoch_dir / f"{split}_states.npz"
                np.savez(
                    save_path,
                    states=[s["states"] for s in self.recorded_states[split]],
                    memory=[s["memory"] for s in self.recorded_states[split]],
                )

        # Clear for next epoch
        self.recorded_states = {"train": [], "val": []}

    def _get_base_model(self, pl_module: pl.LightningModule):
        """Extract base ELM model from task wrapper."""
        if hasattr(pl_module, "base_model"):
            return pl_module.base_model
        return pl_module


class SequenceVisualizationCallback(Callback):
    """
    Visualizes sequence predictions during training.

    Creates plots comparing model predictions with ground truth.
    Supports time series, classification, and regression tasks.

    Example:
        callback = SequenceVisualizationCallback(
            log_every_n_epochs=10,
            num_samples=4,
        )
        trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        num_samples: int = 4,
        task_type: Literal[
            "classification", "regression", "timeseries"
        ] = "classification",
        save_dir: str | None = None,
        log_to_wandb: bool = True,
    ):
        """
        Initialize sequence visualization callback.

        Args:
            log_every_n_epochs: Create visualizations every N epochs
            num_samples: Number of samples to visualize
            task_type: Type of task (classification, regression, timeseries)
            save_dir: Directory to save plots (default: None)
            log_to_wandb: Log to wandb if available
        """
        super().__init__()
        plt.switch_backend('agg')
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.task_type = task_type
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_to_wandb = log_to_wandb

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Create visualizations at end of validation epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        # Handle case where val_dataloaders returns a list
        if isinstance(val_dataloader, list):
            val_dataloader = val_dataloader[0]

        # Get a batch
        batch = next(iter(val_dataloader))
        inputs, targets = batch

        # Move to device
        inputs = inputs.to(pl_module.device)
        # Handle tuple targets (e.g., NeuronIO returns (spike_targets, soma_targets))
        if isinstance(targets, (tuple, list)):
            targets = tuple(t.to(pl_module.device) for t in targets)
        else:
            targets = targets.to(pl_module.device)

        # Get predictions
        with torch.no_grad():
            if hasattr(pl_module, "base_model"):
                outputs = pl_module.base_model(inputs)
            else:
                outputs = pl_module(inputs)

        # Limit to num_samples
        inputs = inputs[: self.num_samples]
        if isinstance(targets, tuple):
            targets = tuple(t[: self.num_samples] for t in targets)
        else:
            targets = targets[: self.num_samples]
        outputs = outputs[: self.num_samples]

        # Create visualization based on task type
        if self.task_type == "timeseries":
            fig = self._plot_timeseries(inputs, targets, outputs)
        elif self.task_type == "regression":
            fig = self._plot_regression(inputs, targets, outputs)
        else:  # classification
            fig = self._plot_classification(inputs, targets, outputs)

        # Save to disk
        if self.save_dir:
            save_path = self.save_dir / f"predictions_epoch_{trainer.current_epoch}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        # Log to wandb
        if self.log_to_wandb and trainer.logger is not None:
            if hasattr(trainer.logger, "experiment"):
                try:
                    import wandb

                    trainer.logger.experiment.log(
                        {
                            "predictions": wandb.Image(fig),
                            "epoch": trainer.current_epoch,
                        }
                    )
                except ImportError:
                    pass

        plt.close(fig)

    def _plot_timeseries(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
    ) -> plt.Figure:
        """Plot time series predictions."""
        num_samples = inputs.shape[0]
        seq_len = outputs.shape[1]

        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            # Plot predictions and targets
            ax.plot(targets[i].cpu().numpy(), label="Target", linewidth=2)
            ax.plot(
                outputs[i].cpu().numpy(), label="Prediction", linewidth=2, alpha=0.7
            )

            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(f"Sample {i+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_regression(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
    ) -> plt.Figure:
        """Plot regression predictions."""
        num_samples = inputs.shape[0]

        # Handle tuple targets (e.g., NeuronIO returns (spike_targets, soma_targets))
        # Use soma targets (second element) for regression visualization
        if isinstance(targets, tuple):
            targets = targets[1]  # soma_targets
            # For NeuronIO: outputs are (batch, time, 2) where last dim is [spike, soma]
            if outputs.dim() == 3 and outputs.shape[-1] == 2:
                outputs = outputs[..., 1]  # soma output (batch, time)
        else:
            # Use last timestep for regression (non-NeuronIO case)
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        targets_np = targets.cpu().numpy().flatten()
        outputs_np = outputs.cpu().numpy().flatten()

        axes[0].scatter(targets_np, outputs_np, alpha=0.6)
        axes[0].plot(
            [targets_np.min(), targets_np.max()],
            [targets_np.min(), targets_np.max()],
            "r--",
            linewidth=2,
            label="Perfect prediction",
        )
        axes[0].set_xlabel("Target")
        axes[0].set_ylabel("Prediction")
        axes[0].set_title("Predictions vs Targets")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = targets_np - outputs_np
        axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Residuals (Mean: {residuals.mean():.4f})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_classification(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
    ) -> plt.Figure:
        """Plot classification predictions."""
        num_samples = inputs.shape[0]

        # Use last timestep for classification
        if outputs.dim() == 3:
            outputs = outputs[:, -1, :]

        # Get predictions
        predictions = outputs.argmax(dim=-1)

        # Create figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Plot input sequence (if it's 2D)
            if inputs.dim() == 3 and inputs.shape[-1] <= 100:
                ax_seq = axes[i, 0]
                ax_seq.imshow(
                    inputs[i].cpu().numpy().T,
                    aspect="auto",
                    cmap="viridis",
                    interpolation="nearest",
                )
                ax_seq.set_xlabel("Time")
                ax_seq.set_ylabel("Feature")
                ax_seq.set_title(f"Sample {i+1}: Input Sequence")

            # Plot class probabilities
            ax_prob = axes[i, 1]
            probs = torch.softmax(outputs[i], dim=-1).cpu().numpy()
            classes = np.arange(len(probs))

            bars = ax_prob.bar(classes, probs)

            # Highlight predicted and true class
            pred_class = predictions[i].item()
            true_class = targets[i].item()

            bars[pred_class].set_color("green" if pred_class == true_class else "red")
            bars[pred_class].set_alpha(0.8)

            ax_prob.axvline(
                true_class,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"True: {true_class}",
            )
            ax_prob.set_xlabel("Class")
            ax_prob.set_ylabel("Probability")
            ax_prob.set_title(f"Predicted: {pred_class} | True: {true_class}")
            ax_prob.legend()
            ax_prob.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig


class MemoryDynamicsCallback(Callback):
    """
    Visualizes memory unit dynamics over time.

    Tracks how memory units evolve during sequence processing.
    Useful for understanding long-range dependencies and memory usage.

    Example:
        callback = MemoryDynamicsCallback(
            log_every_n_epochs=10,
            num_samples=2,
        )
        trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        num_samples: int = 2,
        save_dir: str | None = None,
        log_to_wandb: bool = True,
    ):
        """
        Initialize memory dynamics callback.

        Args:
            log_every_n_epochs: Create visualizations every N epochs
            num_samples: Number of samples to visualize
            save_dir: Directory to save plots
            log_to_wandb: Log to wandb if available
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_to_wandb = log_to_wandb

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Create memory dynamics visualizations."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get base model
        base_model = self._get_base_model(pl_module)
        if base_model is None or not hasattr(base_model, "forward_with_states"):
            return

        # Get validation batch
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))
        inputs, _ = batch
        inputs = inputs[: self.num_samples].to(pl_module.device)

        # Get memory dynamics
        with torch.no_grad():
            _, _, memory = base_model.forward_with_states(inputs)

        # Create visualization
        fig = self._plot_memory_dynamics(memory)

        # Save
        if self.save_dir:
            save_path = self.save_dir / f"memory_epoch_{trainer.current_epoch}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        # Log to wandb
        if self.log_to_wandb and trainer.logger is not None:
            if hasattr(trainer.logger, "experiment"):
                try:
                    import wandb

                    trainer.logger.experiment.log(
                        {
                            "memory_dynamics": wandb.Image(fig),
                            "epoch": trainer.current_epoch,
                        }
                    )
                except ImportError:
                    pass

        plt.close(fig)

    def _plot_memory_dynamics(self, memory: torch.Tensor) -> plt.Figure:
        """
        Plot memory unit activations over time.

        Args:
            memory: (batch, time, num_memory) tensor

        Returns:
            Matplotlib figure
        """
        memory_np = memory.cpu().numpy()
        num_samples, seq_len, num_memory = memory_np.shape

        fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Heatmap of memory activations
            ax_heat = axes[i, 0]
            im = ax_heat.imshow(
                memory_np[i].T,
                aspect="auto",
                cmap="RdBu_r",
                interpolation="nearest",
                vmin=-memory_np[i].max(),
                vmax=memory_np[i].max(),
            )
            ax_heat.set_xlabel("Time")
            ax_heat.set_ylabel("Memory Unit")
            ax_heat.set_title(f"Sample {i+1}: Memory Activations")
            plt.colorbar(im, ax=ax_heat)

            # Line plots for individual memory units
            ax_line = axes[i, 1]

            # Plot a subset of memory units
            num_units_to_plot = min(10, num_memory)
            unit_indices = np.linspace(0, num_memory - 1, num_units_to_plot, dtype=int)

            for unit_idx in unit_indices:
                ax_line.plot(
                    memory_np[i, :, unit_idx],
                    label=f"Unit {unit_idx}",
                    alpha=0.7,
                )

            ax_line.set_xlabel("Time")
            ax_line.set_ylabel("Activation")
            ax_line.set_title(f"Sample {i+1}: Memory Unit Trajectories")
            ax_line.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax_line.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _get_base_model(self, pl_module: pl.LightningModule):
        """Extract base ELM model from task wrapper."""
        if hasattr(pl_module, "base_model"):
            return pl_module.base_model
        return pl_module
