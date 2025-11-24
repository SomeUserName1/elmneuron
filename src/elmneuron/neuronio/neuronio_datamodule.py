"""
NeuronIO LightningDataModule.

This module provides a PyTorch Lightning DataModule wrapper for the
NeuronIO biophysical neuron dataset, with integrated routing transforms.
"""

from typing import Any, Literal

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..transforms import NeuronIORouting, RoutingTransform
from .neuronio_data_loader import NeuronIO
from .neuronio_data_utils import (
    DEFAULT_Y_SOMA_THRESHOLD,
    DEFAULT_Y_TRAIN_SOMA_BIAS,
    DEFAULT_Y_TRAIN_SOMA_SCALE,
    NEURONIO_DATA_DIM,
    NEURONIO_LABEL_DIM,
    NEURONIO_SIM_LEN,
    NEURONIO_SIM_PER_FILE,
    create_neuronio_input_type,
    get_data_files_from_folder,
)


class _RoutingCollate:
    """Collate function that applies routing transform."""

    def __init__(self, routing: RoutingTransform | None):
        self.routing = routing

    def __call__(self, batch):
        X_batch, (y_spike_batch, y_soma_batch) = batch

        # Apply routing transform to input data
        if self.routing is not None:
            X_batch = self.routing(X_batch)

        return X_batch, (y_spike_batch, y_soma_batch)


class NeuronIODataModule(pl.LightningDataModule):
    """
    Lightning DataModule for NeuronIO biophysical neuron dataset.

    This DataModule wraps the existing NeuronIO IterableDataset and
    provides train/val/test splits with optional routing transforms.
    """

    def __init__(
        self,
        data_folders: list[str] | None = None,
        train_files: list[str] | None = None,
        val_files: list[str] | None = None,
        test_files: list[str] | None = None,
        routing: RoutingTransform | None = None,
        batch_size: int = 8,
        input_window_size: int = 500,
        file_load_fraction: float = 0.3,
        ignore_time_from_start: int = 150,
        num_workers: int = 3,
        num_prefetch_batch: int = 5,
        y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
        y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
        y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
        neuronio_sim_per_file: int = NEURONIO_SIM_PER_FILE,
        neuronio_sim_len: int = NEURONIO_SIM_LEN,
        neuronio_label_dim: int = NEURONIO_LABEL_DIM,
        neuronio_data_dim: int = NEURONIO_DATA_DIM,
        train_batches_per_epoch: int = 500,
        val_batches_per_epoch: int = 100,
        test_batches_per_epoch: int = 200,
        seed: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize NeuronIO DataModule.

        Args:
            data_folders: List of folders containing data files (for train/val split)
            train_files: Explicit list of training files (overrides data_folders)
            val_files: Explicit list of validation files
            test_files: Explicit list of test files
            routing: Optional routing transform to apply to inputs
            batch_size: Batch size (default: 8)
            input_window_size: Length of temporal windows (default: 500)
            file_load_fraction: Fraction of file to use per load (default: 0.3)
            ignore_time_from_start: Ignore initial timesteps (default: 500)
            num_workers: Number of worker processes (default: 5)
            num_prefetch_batch: Number of batches to prefetch (default: 50)
            y_soma_threshold: Soma voltage threshold (default: -55.0)
            y_train_soma_bias: Soma voltage bias (default: -67.7)
            y_train_soma_scale: Soma voltage scale (default: 0.1)
            neuronio_sim_per_file: Simulations per file (default: 128)
            neuronio_sim_len: Simulation length (default: 6000)
            neuronio_label_dim: Label dimension (default: 2)
            neuronio_data_dim: Data dimension (default: 1278)
            train_batches_per_epoch: Training batches per epoch (default: 500)
            val_batches_per_epoch: Validation batches per epoch (default: 100)
            test_batches_per_epoch: Test batches per epoch (default: 200)
            seed: Random seed (default: 0)
            verbose: Verbose logging (default: False)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["routing"])

        # Store configuration
        self.data_folders = data_folders
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.routing = routing
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.file_load_fraction = file_load_fraction
        self.ignore_time_from_start = ignore_time_from_start
        self.num_workers = num_workers
        self.num_prefetch_batch = num_prefetch_batch
        self.y_soma_threshold = y_soma_threshold
        self.y_train_soma_bias = y_train_soma_bias
        self.y_train_soma_scale = y_train_soma_scale
        self.neuronio_sim_per_file = neuronio_sim_per_file
        self.neuronio_sim_len = neuronio_sim_len
        self.neuronio_label_dim = neuronio_label_dim
        self.neuronio_data_dim = neuronio_data_dim
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        self.test_batches_per_epoch = test_batches_per_epoch
        self.seed = seed
        self.verbose = verbose

        # Create synapse types (excitatory/inhibitory markers)
        # These are always in original dimension (1278) - routing is applied after
        self.synapse_types = create_neuronio_input_type()

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Create collate function for routing
        self._collate_fn = _RoutingCollate(self.routing)

    def prepare_data(self) -> None:
        """
        Download or prepare data (if needed).

        For NeuronIO, data must be manually downloaded from Kaggle.
        This method can be used to validate that data exists.
        """
        # NeuronIO data is too large for automatic download
        # Users must manually download from Kaggle
        if self.data_folders is None and self.train_files is None:
            raise ValueError(
                "NeuronIO data not found. Please download manually from Kaggle:\n"
                "Train: https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data\n"
                "Test: https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data"
            )

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or None (all stages)
        """
        # Get file lists
        if self.train_files is None and self.data_folders is not None:
            all_files = get_data_files_from_folder(self.data_folders)
            # Simple split: first 80% train, next 10% val, last 10% test
            n_files = len(all_files)
            train_end = int(0.8 * n_files)
            val_end = int(0.9 * n_files)

            self.train_files = all_files[:train_end]
            self.val_files = all_files[train_end:val_end]
            self.test_files = all_files[val_end:]

            if self.verbose:
                print(
                    f"Split {n_files} files: {len(self.train_files)} train, "
                    f"{len(self.val_files)} val, {len(self.test_files)} test"
                )

        # Setup for training/validation
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                if self.verbose:
                    print("  Creating train NeuronIO dataset...", flush=True)
                self.train_dataset = NeuronIO(
                    batches_per_epoch=self.train_batches_per_epoch,
                    file_paths=self.train_files,
                    synapse_types=self.synapse_types,
                    batch_size=self.batch_size,
                    input_window_size=self.input_window_size,
                    file_load_fraction=self.file_load_fraction,
                    ignore_time_from_start=self.ignore_time_from_start,
                    y_soma_threshold=self.y_soma_threshold,
                    y_train_soma_bias=self.y_train_soma_bias,
                    y_train_soma_scale=self.y_train_soma_scale,
                    neuronio_sim_per_file=self.neuronio_sim_per_file,
                    neuronio_sim_len=self.neuronio_sim_len,
                    neuronio_label_dim=self.neuronio_label_dim,
                    neuronio_data_dim=self.neuronio_data_dim,
                    seed=self.seed,
                    verbose=self.verbose,
                )
                if self.verbose:
                    print("  Train dataset created successfully", flush=True)

            if self.val_dataset is None and self.val_files is not None:
                if self.verbose:
                    print("  Creating val NeuronIO dataset...", flush=True)
                self.val_dataset = NeuronIO(
                    batches_per_epoch=self.val_batches_per_epoch,
                    file_paths=self.val_files,
                    synapse_types=self.synapse_types,
                    batch_size=self.batch_size,
                    input_window_size=self.input_window_size,
                    file_load_fraction=self.file_load_fraction,
                    ignore_time_from_start=self.ignore_time_from_start,
                    y_soma_threshold=self.y_soma_threshold,
                    y_train_soma_bias=self.y_train_soma_bias,
                    y_train_soma_scale=self.y_train_soma_scale,
                    neuronio_sim_per_file=self.neuronio_sim_per_file,
                    neuronio_sim_len=self.neuronio_sim_len,
                    neuronio_label_dim=self.neuronio_label_dim,
                    neuronio_data_dim=self.neuronio_data_dim,
                    seed=self.seed + 1,  # Different seed for validation
                    verbose=self.verbose,
                )
                if self.verbose:
                    print("  Val dataset created successfully", flush=True)

        # Setup for testing
        if stage == "test" or stage is None:
            if self.test_dataset is None and self.test_files is not None:
                self.test_dataset = NeuronIO(
                    batches_per_epoch=self.test_batches_per_epoch,
                    file_paths=self.test_files,
                    synapse_types=self.synapse_types,
                    batch_size=self.batch_size,
                    input_window_size=self.input_window_size,
                    file_load_fraction=self.file_load_fraction,
                    ignore_time_from_start=self.ignore_time_from_start,
                    y_soma_threshold=self.y_soma_threshold,
                    y_train_soma_bias=self.y_train_soma_bias,
                    y_train_soma_scale=self.y_train_soma_scale,
                    neuronio_sim_per_file=self.neuronio_sim_per_file,
                    neuronio_sim_len=self.neuronio_sim_len,
                    neuronio_label_dim=self.neuronio_label_dim,
                    neuronio_data_dim=self.neuronio_data_dim,
                    seed=self.seed + 2,  # Different seed for test
                    verbose=self.verbose,
                )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        # NeuronIO handles batching internally, so we don't use DataLoader batching
        # DataLoader handles multiprocessing, prefetching, and pin_memory
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Already batched by NeuronIO
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.num_prefetch_batch // max(1, self.num_workers),
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not configured")
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.num_prefetch_batch // max(1, self.num_workers),
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not configured")
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.num_prefetch_batch // max(1, self.num_workers),
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def teardown(self, stage: str | None = None) -> None:
        """
        Clean up datasets.

        Args:
            stage: Current stage
        """
        # Ensure worker processes are terminated
        if self.train_dataset is not None:
            del self.train_dataset
            self.train_dataset = None

        if self.val_dataset is not None:
            del self.val_dataset
            self.val_dataset = None

        if self.test_dataset is not None:
            del self.test_dataset
            self.test_dataset = None

    @property
    def num_classes(self) -> int:
        """Number of output classes/channels."""
        return self.neuronio_label_dim

    @property
    def input_dim(self) -> int:
        """Input dimension after routing."""
        if self.routing is not None:
            return self.routing.num_synapse
        return self.neuronio_data_dim

    def get_routing_info(self) -> dict | None:
        """Get routing configuration info."""
        if self.routing is not None:
            return self.routing.get_routing_info()
        return None
