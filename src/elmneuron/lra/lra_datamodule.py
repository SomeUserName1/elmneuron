"""
Long Range Arena (LRA) DataModules.

Implements PyTorch Lightning DataModules for the LRA benchmark tasks:
- ListOps: Hierarchical reasoning with list operations (2K sequences)
- Text: Byte-level text classification (4K sequences)
- Retrieval: Document matching (8K sequences)
- Image: CIFAR-10 as pixel sequences (1K sequences)
- Pathfinder: Path connectivity detection (1K sequences)

Reference: https://arxiv.org/abs/2011.04006
"""

import gzip
import pickle
import tarfile
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# LRA dataset URL
LRA_RELEASE_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"


def download_lra_dataset(cache_dir: Path) -> None:
    """
    Download and extract LRA dataset.

    Args:
        cache_dir: Directory to store the dataset
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download gzipped tar file
    tar_gz_path = cache_dir / "lra_release.tar.gz"

    if not tar_gz_path.exists():
        print(f"Downloading LRA dataset from {LRA_RELEASE_URL}...")
        print("This may take a while (~7GB download)...")
        urlretrieve(LRA_RELEASE_URL, tar_gz_path)
        print("Download complete!")

    # Extract if not already extracted
    lra_dir = cache_dir / "lra_release"
    if not lra_dir.exists():
        print("Extracting LRA dataset...")
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(cache_dir)
        print("Extraction complete!")


class LRADataset(Dataset):
    """
    Generic LRA dataset wrapper.

    Loads preprocessed LRA data from pickle files.
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "val", "test"] = "train",
    ):
        """
        Initialize LRA dataset.

        Args:
            data_path: Path to the pickle file
            split: Data split (train/val/test)
        """
        self.data_path = data_path
        self.split = split

        # Load data
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # Extract split
        if split == "train":
            self.inputs = data["x_train"]
            self.targets = data["y_train"]
        elif split == "val":
            self.inputs = data["x_val"]
            self.targets = data["y_val"]
        elif split == "test":
            self.inputs = data["x_test"]
            self.targets = data["y_test"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Convert to tensors
        self.inputs = torch.from_numpy(self.inputs).long()
        self.targets = torch.from_numpy(self.targets).long()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item.

        Returns:
            tuple of (input_sequence, target)
        """
        return self.inputs[idx], self.targets[idx]


class BaseLRADataModule(pl.LightningDataModule):
    """
    Base class for LRA DataModules.

    Provides common functionality for all LRA tasks.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
    ):
        """
        Initialize base LRA DataModule.

        Args:
            data_dir: Data directory (default: ~/.cache/elmneuron/lra)
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for GPU transfer
            download: Download dataset if not present
        """
        super().__init__()
        self.save_hyperparameters()

        # Store configuration
        if data_dir is None:
            data_dir = Path.home() / ".cache" / "elmneuron" / "lra"
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Task-specific properties (set by subclasses)
        self.task_name = None
        self.num_classes = None
        self.sequence_length = None
        self.vocab_size = None

    def prepare_data(self) -> None:
        """Download LRA dataset if needed."""
        if self.download:
            download_lra_dataset(self.data_dir)

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

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "task_name": self.task_name,
            "num_classes": self.num_classes,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "batch_size": self.batch_size,
        }


class ListOpsDataModule(BaseLRADataModule):
    """
    ListOps task from Long Range Arena.

    Hierarchical reasoning with nested list operations (MAX, MIN, MEDIAN, SUM_MOD).
    - Sequence length: 2048
    - Vocabulary size: ~20 (operators + numbers + delimiters)
    - Number of classes: 10 (output values 0-9)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
    ):
        """Initialize ListOps DataModule."""
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            download=download,
        )

        self.task_name = "listops"
        self.num_classes = 10
        self.sequence_length = 2048
        self.vocab_size = 18  # Approximate

    def setup(self, stage: str | None = None) -> None:
        """Setup ListOps datasets."""
        lra_dir = self.data_dir / "lra_release" / "lra_release" / "listops-1000"
        data_path = lra_dir / "basic_train.pkl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"ListOps data not found at {data_path}. "
                f"Set download=True to download the LRA dataset."
            )

        if stage == "fit" or stage is None:
            self.train_dataset = LRADataset(data_path, split="train")
            self.val_dataset = LRADataset(data_path, split="val")

        if stage == "test" or stage is None:
            self.test_dataset = LRADataset(data_path, split="test")


class LRATextDataModule(BaseLRADataModule):
    """
    Text Classification task from Long Range Arena.

    Byte-level IMDb sentiment classification.
    - Sequence length: 4096
    - Vocabulary size: 256 (byte-level)
    - Number of classes: 2 (positive/negative sentiment)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
    ):
        """Initialize LRA Text DataModule."""
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            download=download,
        )

        self.task_name = "text"
        self.num_classes = 2
        self.sequence_length = 4096
        self.vocab_size = 256  # Byte-level

    def setup(self, stage: str | None = None) -> None:
        """Setup text classification datasets."""
        lra_dir = self.data_dir / "lra_release" / "lra_release" / "tsv_data"
        data_path = lra_dir / "imdb.train.pkl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Text data not found at {data_path}. "
                f"Set download=True to download the LRA dataset."
            )

        if stage == "fit" or stage is None:
            self.train_dataset = LRADataset(data_path, split="train")
            self.val_dataset = LRADataset(data_path, split="val")

        if stage == "test" or stage is None:
            # Test data is in a separate file
            test_path = lra_dir / "imdb.test.pkl"
            self.test_dataset = LRADataset(test_path, split="test")


class LRARetrievalDataModule(BaseLRADataModule):
    """
    Document Retrieval task from Long Range Arena.

    Match pairs of research paper abstracts (ACL Anthology Network).
    - Sequence length: 4096 per document (8192 total for pair)
    - Vocabulary size: 256 (byte-level)
    - Number of classes: 2 (match/no-match)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        batch_size: int = 16,  # Smaller batch for longer sequences
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
    ):
        """Initialize LRA Retrieval DataModule."""
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            download=download,
        )

        self.task_name = "retrieval"
        self.num_classes = 2
        self.sequence_length = 8192  # Total (4096 × 2)
        self.vocab_size = 256  # Byte-level

    def setup(self, stage: str | None = None) -> None:
        """Setup retrieval datasets."""
        lra_dir = self.data_dir / "lra_release" / "lra_release" / "tsv_data"
        data_path = lra_dir / "aan.train.pkl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Retrieval data not found at {data_path}. "
                f"Set download=True to download the LRA dataset."
            )

        if stage == "fit" or stage is None:
            self.train_dataset = LRADataset(data_path, split="train")
            self.val_dataset = LRADataset(data_path, split="val")

        if stage == "test" or stage is None:
            test_path = lra_dir / "aan.test.pkl"
            self.test_dataset = LRADataset(test_path, split="test")


class LRAImageDataModule(BaseLRADataModule):
    """
    Image Classification task from Long Range Arena.

    CIFAR-10 images treated as pixel sequences (grayscale, flattened).
    - Sequence length: 1024 (32×32 grayscale images)
    - Vocabulary size: 256 (pixel values 0-255)
    - Number of classes: 10 (CIFAR-10 classes)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
    ):
        """Initialize LRA Image DataModule."""
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            download=download,
        )

        self.task_name = "image"
        self.num_classes = 10
        self.sequence_length = 1024  # 32×32 grayscale
        self.vocab_size = 256  # Pixel values

    def setup(self, stage: str | None = None) -> None:
        """Setup image classification datasets."""
        lra_dir = self.data_dir / "lra_release" / "lra_release" / "cifar10"
        data_path = lra_dir / "cifar10.train.pkl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Image data not found at {data_path}. "
                f"Set download=True to download the LRA dataset."
            )

        if stage == "fit" or stage is None:
            self.train_dataset = LRADataset(data_path, split="train")
            self.val_dataset = LRADataset(data_path, split="val")

        if stage == "test" or stage is None:
            test_path = lra_dir / "cifar10.test.pkl"
            self.test_dataset = LRADataset(test_path, split="test")


class LRAPathfinderDataModule(BaseLRADataModule):
    """
    Pathfinder task from Long Range Arena.

    Visual reasoning: determine if two points are connected by a dashed line.
    - Sequence length: 1024 (32×32 grayscale images)
    - Vocabulary size: 256 (pixel values)
    - Number of classes: 2 (connected/not connected)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
    ):
        """
        Initialize LRA Pathfinder DataModule.

        Args:
            data_dir: Data directory
            difficulty: Pathfinder difficulty level
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            download: Download dataset if not present
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            download=download,
        )

        self.task_name = "pathfinder"
        self.difficulty = difficulty
        self.num_classes = 2
        self.sequence_length = 1024  # 32×32 images
        self.vocab_size = 256  # Pixel values

    def setup(self, stage: str | None = None) -> None:
        """Setup pathfinder datasets."""
        lra_dir = self.data_dir / "lra_release" / "lra_release" / "pathfinder32"

        # Pathfinder has difficulty variants
        difficulty_map = {
            "easy": "curv_baseline",
            "medium": "curv_contour_length_9",
            "hard": "curv_contour_length_14",
        }

        subdir = difficulty_map[self.difficulty]
        data_path = lra_dir / subdir / f"metadata.pkl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Pathfinder data not found at {data_path}. "
                f"Set download=True to download the LRA dataset."
            )

        if stage == "fit" or stage is None:
            self.train_dataset = LRADataset(data_path, split="train")
            self.val_dataset = LRADataset(data_path, split="val")

        if stage == "test" or stage is None:
            self.test_dataset = LRADataset(data_path, split="test")
