"""
Dataset modules for ELM models.

This module provides Lightning DataModules for various standard ML datasets,
with support for sequentialization strategies to convert non-sequential data
(like images and text) into sequences for ELM processing.
"""

from .text_datamodule import (
    CustomTextDataModule,
    WikiText2DataModule,
    WikiText103DataModule,
)
from .vision_datamodule import (
    CIFAR10SequenceDataModule,
    CIFAR100SequenceDataModule,
    FashionMNISTSequenceDataModule,
    MNISTSequenceDataModule,
)

__all__ = [
    # Vision datasets
    "MNISTSequenceDataModule",
    "FashionMNISTSequenceDataModule",
    "CIFAR10SequenceDataModule",
    "CIFAR100SequenceDataModule",
    # Text datasets
    "WikiText2DataModule",
    "WikiText103DataModule",
    "CustomTextDataModule",
]
