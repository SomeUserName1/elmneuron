"""
Spiking Heidelberg Digits (SHD) dataset utilities for the ELM neuron.

This module provides data loading and download utilities for the SHD dataset,
including the SHD-Adding variant for testing long-range dependencies.
"""

from elmneuron.shd.shd_data_loader import (
    SHD,
    SHDAdding,
    random_val_split_SHD_data,
    visualize_training_batch,
)
from elmneuron.shd.shd_datamodule import SHDDataModule
from elmneuron.shd.shd_download_utils import get_shd_dataset

__all__ = [
    # Datasets
    "SHD",
    "SHDAdding",
    "SHDDataModule",
    # Utilities
    "random_val_split_SHD_data",
    "visualize_training_batch",
    "get_shd_dataset",
]
