"""
Vision dataset DataModules with sequentialization support.

This module provides Lightning DataModules for standard vision datasets
(MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100) with support for converting
images to sequences using various strategies.
"""

from pathlib import Path
from typing import Callable, Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchvision import datasets, transforms

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print(
        "Warning: torchvision not available. Install with: pip install elmneuron[vision]"
    )

from ..transforms import PatchSequence, PixelSequence, SequenceTransform


class SequentialDatasetWrapper(Dataset):
    """
    Wrapper that applies sequentialization transform to a dataset.

    Converts images (C, H, W) to sequences (T, D) where:
    - T is sequence length (depends on sequentialization strategy)
    - D is feature dimension per timestep
    """

    def __init__(
        self,
        base_dataset: Dataset,
        sequence_transform: SequenceTransform,
        normalize: bool = True,
    ):
        """
        Initialize sequential dataset wrapper.

        Args:
            base_dataset: Base vision dataset (returns images)
            sequence_transform: Transform to convert images to sequences
            normalize: Whether to normalize images before sequentialization
        """
        self.base_dataset = base_dataset
        self.sequence_transform = sequence_transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get sequential data item.

        Args:
            idx: Index

        Returns:
            tuple of (sequence, label):
            - sequence: (time, features) tensor
            - label: class label
        """
        image, label = self.base_dataset[idx]

        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        # Normalize if requested
        if self.normalize and image.dtype == torch.uint8:
            image = image.float() / 255.0

        # Add batch dimension, apply transform, remove batch dimension
        sequence = self.sequence_transform(image.unsqueeze(0)).squeeze(0)

        return sequence, label


class BaseVisionSequenceDataModule(pl.LightningDataModule):
    """
    Base class for vision dataset DataModules with sequentialization.

    Provides common functionality for MNIST, CIFAR, etc.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        sequence_transform: SequenceTransform | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        Initialize base vision DataModule.

        Args:
            data_dir: Data directory (default: ~/.cache/elmneuron/data)
            sequence_transform: Transform to convert images to sequences
            batch_size: Batch size (default: 32)
            num_workers: Number of data loading workers (default: 4)
            pin_memory: Pin memory for GPU transfer (default: True)
            val_split: Validation split fraction (default: 0.1)
            normalize: Normalize images (default: True)
            seed: Random seed (default: 42)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["sequence_transform"])

        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for vision datasets. "
                "Install with: pip install elmneuron[vision]"
            )

        # Store configuration
        if data_dir is None:
            data_dir = Path.home() / ".cache" / "elmneuron" / "data"
        self.data_dir = Path(data_dir)
        self.sequence_transform = sequence_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.normalize = normalize
        self.seed = seed

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Dataset-specific properties (set by subclasses)
        self.num_classes = None
        self.image_shape = None  # (C, H, W)

    def prepare_data(self) -> None:
        """Download data if needed. Called on single process."""
        # Implemented by subclasses
        pass

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage."""
        # Implemented by subclasses
        pass

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    @property
    def input_dim(self) -> int:
        """Input dimension per timestep after sequentialization."""
        if self.sequence_transform is None:
            # Default: flatten image
            c, h, w = self.image_shape
            return c * h * w

        # Get dimension from transform
        # Create dummy image and apply transform
        dummy_img = torch.randn(1, *self.image_shape)
        dummy_seq = self.sequence_transform(dummy_img)
        return dummy_seq.shape[-1]

    @property
    def sequence_length(self) -> int:
        """Sequence length after sequentialization."""
        if self.sequence_transform is None:
            return 1  # Single timestep (flattened image)

        # Get sequence length from transform
        dummy_img = torch.randn(1, *self.image_shape)
        dummy_seq = self.sequence_transform(dummy_img)
        return dummy_seq.shape[1]

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "num_classes": self.num_classes,
            "image_shape": self.image_shape,
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }


class MNISTSequenceDataModule(BaseVisionSequenceDataModule):
    """
    MNIST dataset with sequentialization support.

    Converts 28x28 grayscale images to sequences using various strategies.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        sequence_transform: SequenceTransform | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        Initialize MNIST DataModule.

        Args:
            data_dir: Data directory
            sequence_transform: Transform to convert images to sequences
                If None, uses PatchSequence(patch_size=7) by default
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            val_split: Validation split fraction
            normalize: Normalize images
            seed: Random seed
        """
        # Default sequentialization: 4x4 patches (16 patches, 49 dims each)
        if sequence_transform is None:
            sequence_transform = PatchSequence(
                patch_size=7, order="raster", flatten=True
            )

        super().__init__(
            data_dir=data_dir,
            sequence_transform=sequence_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            normalize=normalize,
            seed=seed,
        )

        self.num_classes = 10
        self.image_shape = (1, 28, 28)  # Grayscale

    def prepare_data(self) -> None:
        """Download MNIST data."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Setup MNIST datasets."""
        # Basic transforms (to tensor)
        transform = transforms.ToTensor()

        if stage == "fit" or stage is None:
            # Load full training set
            full_train = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=transform,
            )

            # Split into train/val
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size

            train_subset, val_subset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Wrap with sequentialization
            self.train_dataset = SequentialDatasetWrapper(
                train_subset, self.sequence_transform, self.normalize
            )
            self.val_dataset = SequentialDatasetWrapper(
                val_subset, self.sequence_transform, self.normalize
            )

        if stage == "test" or stage is None:
            test_set = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=transform,
            )
            self.test_dataset = SequentialDatasetWrapper(
                test_set, self.sequence_transform, self.normalize
            )


class FashionMNISTSequenceDataModule(MNISTSequenceDataModule):
    """
    Fashion-MNIST dataset with sequentialization support.

    Same structure as MNIST but with fashion items instead of digits.
    """

    def prepare_data(self) -> None:
        """Download Fashion-MNIST data."""
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Setup Fashion-MNIST datasets."""
        transform = transforms.ToTensor()

        if stage == "fit" or stage is None:
            full_train = datasets.FashionMNIST(
                self.data_dir,
                train=True,
                transform=transform,
            )

            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size

            train_subset, val_subset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = SequentialDatasetWrapper(
                train_subset, self.sequence_transform, self.normalize
            )
            self.val_dataset = SequentialDatasetWrapper(
                val_subset, self.sequence_transform, self.normalize
            )

        if stage == "test" or stage is None:
            test_set = datasets.FashionMNIST(
                self.data_dir,
                train=False,
                transform=transform,
            )
            self.test_dataset = SequentialDatasetWrapper(
                test_set, self.sequence_transform, self.normalize
            )


class CIFAR10SequenceDataModule(BaseVisionSequenceDataModule):
    """
    CIFAR-10 dataset with sequentialization support.

    Converts 32x32 RGB images to sequences.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        sequence_transform: SequenceTransform | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        normalize: bool = True,
        data_augmentation: bool = True,
        seed: int = 42,
    ):
        """
        Initialize CIFAR-10 DataModule.

        Args:
            data_dir: Data directory
            sequence_transform: Transform to convert images to sequences
                If None, uses PatchSequence(patch_size=8) by default
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            val_split: Validation split fraction
            normalize: Normalize images
            data_augmentation: Apply data augmentation (flip, crop)
            seed: Random seed
        """
        # Default: 4x4 patches (16 patches, 192 dims each)
        if sequence_transform is None:
            sequence_transform = PatchSequence(
                patch_size=8, order="raster", flatten=True
            )

        super().__init__(
            data_dir=data_dir,
            sequence_transform=sequence_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            normalize=normalize,
            seed=seed,
        )

        self.num_classes = 10
        self.image_shape = (3, 32, 32)  # RGB
        self.data_augmentation = data_augmentation

    def prepare_data(self) -> None:
        """Download CIFAR-10 data."""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Setup CIFAR-10 datasets."""
        # Basic transform
        if self.data_augmentation and (stage == "fit" or stage is None):
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform = transforms.ToTensor()

        test_transform = transforms.ToTensor()

        if stage == "fit" or stage is None:
            full_train = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=train_transform,
            )

            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size

            train_subset, val_subset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = SequentialDatasetWrapper(
                train_subset, self.sequence_transform, self.normalize
            )
            self.val_dataset = SequentialDatasetWrapper(
                val_subset, self.sequence_transform, self.normalize
            )

        if stage == "test" or stage is None:
            test_set = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=test_transform,
            )
            self.test_dataset = SequentialDatasetWrapper(
                test_set, self.sequence_transform, self.normalize
            )


class CIFAR100SequenceDataModule(CIFAR10SequenceDataModule):
    """
    CIFAR-100 dataset with sequentialization support.

    Same structure as CIFAR-10 but with 100 classes.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        sequence_transform: SequenceTransform | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        normalize: bool = True,
        data_augmentation: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            sequence_transform=sequence_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            normalize=normalize,
            data_augmentation=data_augmentation,
            seed=seed,
        )
        self.num_classes = 100  # Override for CIFAR-100

    def prepare_data(self) -> None:
        """Download CIFAR-100 data."""
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Setup CIFAR-100 datasets."""
        if self.data_augmentation and (stage == "fit" or stage is None):
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform = transforms.ToTensor()

        test_transform = transforms.ToTensor()

        if stage == "fit" or stage is None:
            full_train = datasets.CIFAR100(
                self.data_dir,
                train=True,
                transform=train_transform,
            )

            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size

            train_subset, val_subset = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = SequentialDatasetWrapper(
                train_subset, self.sequence_transform, self.normalize
            )
            self.val_dataset = SequentialDatasetWrapper(
                val_subset, self.sequence_transform, self.normalize
            )

        if stage == "test" or stage is None:
            test_set = datasets.CIFAR100(
                self.data_dir,
                train=False,
                transform=test_transform,
            )
            self.test_dataset = SequentialDatasetWrapper(
                test_set, self.sequence_transform, self.normalize
            )
