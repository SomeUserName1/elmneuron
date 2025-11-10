"""
Spiking Heidelberg Digits (SHD) LightningDataModule.

This module provides a PyTorch Lightning DataModule wrapper for the
SHD dataset and the SHD-Adding dataset (our proposed variant).
"""

import os
from pathlib import Path
from typing import Literal

import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .shd_data_loader import SHD, SHDAdding, random_val_split_SHD_data
from .shd_download_utils import get_shd_dataset


class SHDDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Spiking Heidelberg Digits (SHD) dataset.

    The SHD dataset contains spike trains from a cochlear model processing
    spoken digits (0-9 in English and German), resulting in 20 classes.

    The dataset automatically downloads from zenkelab.org if not present.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        dataset_variant: Literal["shd", "shd_adding"] = "shd",
        batch_size: int = 256,
        bin_size: int = 25,
        val_fraction: float = 0.1,
        batches_per_epoch: int = 500,
        seed: int = 0,
    ):
        """
        Initialize SHD DataModule.

        Args:
            data_dir: Directory for dataset storage (default: ~/.data-cache/shd)
            dataset_variant: 'shd' for classification or 'shd_adding' for adding task
            batch_size: Batch size (default: 256)
            bin_size: Time bin size in milliseconds (default: 25)
            val_fraction: Fraction of training data for validation (default: 0.1)
            batches_per_epoch: Batches per epoch for SHD-Adding (default: 500)
            seed: Random seed (default: 0)
        """
        super().__init__()
        self.save_hyperparameters()

        # Store configuration
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser("~"), ".data-cache")
        self.data_dir = Path(data_dir)
        self.cache_subdir = "shd"
        self.dataset_variant = dataset_variant
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.val_fraction = val_fraction
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed

        # Dataset properties
        self.num_input_channels = 700
        if dataset_variant == "shd":
            self.num_classes = 20  # 10 English + 10 German digits
        else:  # shd_adding
            self.num_classes = 19  # Sum of two digits (0-18)

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # HDF5 file handles
        self.train_file = None
        self.test_file = None

    def prepare_data(self) -> None:
        """
        Download SHD dataset if not present.

        This is called only on the main process, ensuring thread-safe download.
        """
        # Download dataset if needed
        get_shd_dataset(
            cache_dir=str(self.data_dir),
            cache_subdir=self.cache_subdir,
        )

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or None (all stages)
        """
        # Construct file paths
        data_path = self.data_dir / self.cache_subdir
        train_file_path = data_path / "shd_train.h5"
        test_file_path = data_path / "shd_test.h5"

        # Verify files exist
        if not train_file_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_file_path}. "
                "Run prepare_data() first."
            )
        if not test_file_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {test_file_path}. " "Run prepare_data() first."
            )

        # Open HDF5 files
        if self.train_file is None:
            self.train_file = h5py.File(train_file_path, "r")
        if self.test_file is None:
            self.test_file = h5py.File(test_file_path, "r")

        # Setup for training/validation
        if stage == "fit" or stage is None:
            if self.dataset_variant == "shd":
                # Split training data into train/val
                X_train, y_train, X_val, y_val = random_val_split_SHD_data(
                    self.train_file, self.val_fraction, seed=self.seed
                )

                # Create training dataset
                self.train_dataset = SHD(
                    X=X_train,
                    y=y_train,
                    batch_size=self.batch_size,
                    bin_size=self.bin_size,
                    shuffle=True,
                    test_set=False,
                    seed=self.seed,
                )

                # Create validation dataset
                self.val_dataset = SHD(
                    X=X_val,
                    y=y_val,
                    batch_size=self.batch_size,
                    bin_size=self.bin_size,
                    shuffle=False,
                    test_set=False,
                    seed=self.seed + 1,
                )

            else:  # shd_adding
                # Split training data
                X_train, y_train, X_val, y_val = random_val_split_SHD_data(
                    self.train_file, self.val_fraction, seed=self.seed
                )

                # Create SHD-Adding datasets
                self.train_dataset = SHDAdding(
                    X=X_train,
                    y=y_train,
                    batch_size=self.batch_size,
                    bin_size=self.bin_size,
                    batches_per_epoch=self.batches_per_epoch,
                    shuffle=True,
                    seed=self.seed,
                )

                self.val_dataset = SHDAdding(
                    X=X_val,
                    y=y_val,
                    batch_size=self.batch_size,
                    bin_size=self.bin_size,
                    batches_per_epoch=self.batches_per_epoch // 5,  # Fewer val batches
                    shuffle=False,
                    seed=self.seed + 1,
                )

        # Setup for testing
        if stage == "test" or stage is None:
            if self.dataset_variant == "shd":
                # Test dataset (full SHD test set)
                self.test_dataset = SHD(
                    X=self.test_file["spikes"],
                    y=self.test_file["labels"],
                    batch_size=self.batch_size,
                    bin_size=self.bin_size,
                    shuffle=False,
                    test_set=True,
                    seed=self.seed + 2,
                )
            else:  # shd_adding
                self.test_dataset = SHDAdding(
                    X=self.test_file["spikes"],
                    y=self.test_file["labels"],
                    batch_size=self.batch_size,
                    bin_size=self.bin_size,
                    batches_per_epoch=self.batches_per_epoch // 2,
                    shuffle=False,
                    seed=self.seed + 2,
                )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # SHD handles batching internally
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            num_workers=0,
        )

    def teardown(self, stage: str | None = None) -> None:
        """
        Clean up datasets and close HDF5 files.

        Args:
            stage: Current stage
        """
        # Close HDF5 files
        if self.train_file is not None:
            self.train_file.close()
            self.train_file = None

        if self.test_file is not None:
            self.test_file.close()
            self.test_file = None

        # Clear datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def input_dim(self) -> int:
        """Input dimension (number of cochlear channels)."""
        return self.num_input_channels

    @property
    def sequence_length(self) -> int:
        """Sequence length after binning."""
        if hasattr(self.train_dataset, "num_time_bins"):
            return self.train_dataset.num_time_bins
        # Approximate: 1 second / bin_size (in ms)
        return int(1000 / self.bin_size) + 1

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "dataset_variant": self.dataset_variant,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "bin_size": self.bin_size,
        }
