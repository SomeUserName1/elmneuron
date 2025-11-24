"""
Text dataset DataModules for language modeling and text classification.

This module provides Lightning DataModules for text datasets with support
for various tokenization strategies and sequence chunking.
"""

from pathlib import Path
from typing import Callable, Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchtext.data.utils import get_tokenizer
    from torchtext.datasets import WikiText2, WikiText103

    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False
    print("Warning: torchtext not available. Install with: pip install torchtext")


class TokenizedTextDataset(Dataset):
    """
    Dataset for tokenized text sequences.

    Converts raw text into tokenized sequences with specified length.
    """

    def __init__(
        self,
        text_data: list[str] | str,
        tokenizer: Callable[[str], list[str]] | None = None,
        vocab: dict[str, int] | None = None,
        max_vocab_size: int = 10000,
        sequence_length: int = 128,
        stride: int | None = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        """
        Initialize tokenized text dataset.

        Args:
            text_data: Raw text (string or list of strings)
            tokenizer: Tokenization function (default: character-level)
            vocab: Existing vocabulary mapping
            max_vocab_size: Maximum vocabulary size (default: 10000)
            sequence_length: Length of each sequence (default: 128)
            stride: Stride for sliding window (default: sequence_length)
            unk_token: Unknown token
            pad_token: Padding token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Combine text if list
        if isinstance(text_data, list):
            text_data = " ".join(text_data)

        # Default: character-level tokenization
        if tokenizer is None:
            tokenizer = list  # Split into characters

        # Tokenize text
        tokens = tokenizer(text_data)

        # Build or use vocabulary
        if vocab is None:
            self.vocab = self._build_vocab(tokens, max_vocab_size)
        else:
            self.vocab = vocab

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Convert tokens to indices
        self.token_ids = self._tokens_to_ids(tokens)

        # Create sequences using sliding window
        self.sequences = self._create_sequences()

    def _build_vocab(self, tokens: list[str], max_vocab_size: int) -> dict[str, int]:
        """Build vocabulary from tokens."""
        # Count token frequencies
        from collections import Counter

        token_counts = Counter(tokens)

        # Add special tokens
        vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        # Add most common tokens
        for token, _ in token_counts.most_common(max_vocab_size - 4):
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert tokens to IDs using vocabulary."""
        unk_id = self.vocab[self.unk_token]
        return [self.vocab.get(token, unk_id) for token in tokens]

    def _create_sequences(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Create input-target sequence pairs using sliding window."""
        sequences = []

        # Sliding window over token IDs
        for i in range(0, len(self.token_ids) - self.sequence_length, self.stride):
            # Input: current sequence
            input_seq = self.token_ids[i : i + self.sequence_length]

            # Target: next tokens (shifted by 1)
            target_seq = self.token_ids[i + 1 : i + self.sequence_length + 1]

            # Convert to tensors
            input_tensor = torch.tensor(input_seq, dtype=torch.long)
            target_tensor = torch.tensor(target_seq, dtype=torch.long)

            sequences.append((input_tensor, target_tensor))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sequence pair (input, target)."""
        return self.sequences[idx]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.vocab)


class BaseTextDataModule(pl.LightningDataModule):
    """
    Base class for text dataset DataModules.

    Provides common functionality for WikiText, etc.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        tokenizer_type: Literal["char", "word", "basic"] = "char",
        max_vocab_size: int = 10000,
        sequence_length: int = 128,
        stride: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize base text DataModule.

        Args:
            data_dir: Data directory
            tokenizer_type: Type of tokenizer ('char', 'word', 'basic')
            max_vocab_size: Maximum vocabulary size
            sequence_length: Length of each sequence
            stride: Stride for sliding window (default: sequence_length)
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            val_split: Validation split fraction
            seed: Random seed
        """
        super().__init__()
        self.save_hyperparameters()

        # Store configuration
        if data_dir is None:
            data_dir = Path.home() / ".cache" / "elmneuron" / "data"
        self.data_dir = Path(data_dir)
        self.tokenizer_type = tokenizer_type
        self.max_vocab_size = max_vocab_size
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.seed = seed

        # Create tokenizer
        self.tokenizer = self._create_tokenizer()

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Vocabulary (built during setup)
        self.vocab = None

    def _create_tokenizer(self) -> Callable[[str], list[str]]:
        """Create tokenizer based on type."""
        if self.tokenizer_type == "char":
            # Character-level
            return list
        elif self.tokenizer_type == "word":
            # Simple word tokenization
            return lambda text: text.lower().split()
        elif self.tokenizer_type == "basic":
            # Basic English tokenizer
            if TORCHTEXT_AVAILABLE:
                return get_tokenizer("basic_english")
            else:
                # Fallback to simple split
                return lambda text: text.lower().split()
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def prepare_data(self) -> None:
        """Download data if needed."""
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
    def vocab_size(self) -> int:
        """Vocabulary size."""
        if self.vocab is not None:
            return len(self.vocab)
        return self.max_vocab_size  # Estimate before setup

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "tokenizer_type": self.tokenizer_type,
            "batch_size": self.batch_size,
        }


class WikiText2DataModule(BaseTextDataModule):
    """
    WikiText-2 dataset for language modeling.

    A smaller version of WikiText with ~2M tokens, suitable for quick experiments.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        tokenizer_type: Literal["char", "word", "basic"] = "word",
        max_vocab_size: int = 10000,
        sequence_length: int = 128,
        stride: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        """
        Initialize WikiText-2 DataModule.

        Args:
            data_dir: Data directory
            tokenizer_type: Type of tokenizer
            max_vocab_size: Maximum vocabulary size
            sequence_length: Sequence length
            stride: Stride for sliding window
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory
            seed: Random seed
        """
        super().__init__(
            data_dir=data_dir,
            tokenizer_type=tokenizer_type,
            max_vocab_size=max_vocab_size,
            sequence_length=sequence_length,
            stride=stride,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=0.0,  # WikiText has predefined splits
            seed=seed,
        )

    def prepare_data(self) -> None:
        """Download WikiText-2 data."""
        if not TORCHTEXT_AVAILABLE:
            raise ImportError(
                "torchtext is required for WikiText datasets. "
                "Install with: pip install torchtext"
            )

        # Download train, val, test splits
        WikiText2(self.data_dir, split="train")
        WikiText2(self.data_dir, split="valid")
        WikiText2(self.data_dir, split="test")

    def setup(self, stage: str | None = None) -> None:
        """Setup WikiText-2 datasets."""
        if not TORCHTEXT_AVAILABLE:
            raise ImportError("torchtext is required for WikiText datasets")

        if stage == "fit" or stage is None:
            # Load training data
            train_iter = WikiText2(self.data_dir, split="train")
            train_text = " ".join(train_iter)

            # Build vocabulary from training data
            self.train_dataset = TokenizedTextDataset(
                text_data=train_text,
                tokenizer=self.tokenizer,
                vocab=None,  # Build new vocab
                max_vocab_size=self.max_vocab_size,
                sequence_length=self.sequence_length,
                stride=self.stride,
            )

            # Use same vocab for validation
            self.vocab = self.train_dataset.vocab

            # Load validation data
            val_iter = WikiText2(self.data_dir, split="valid")
            val_text = " ".join(val_iter)

            self.val_dataset = TokenizedTextDataset(
                text_data=val_text,
                tokenizer=self.tokenizer,
                vocab=self.vocab,  # Reuse vocab
                sequence_length=self.sequence_length,
                stride=self.stride,
            )

        if stage == "test" or stage is None:
            # Load test data
            test_iter = WikiText2(self.data_dir, split="test")
            test_text = " ".join(test_iter)

            # Use vocab from training (must call setup('fit') first)
            if self.vocab is None:
                # Build vocab if not already built
                train_iter = WikiText2(self.data_dir, split="train")
                train_text = " ".join(train_iter)
                temp_dataset = TokenizedTextDataset(
                    text_data=train_text,
                    tokenizer=self.tokenizer,
                    max_vocab_size=self.max_vocab_size,
                    sequence_length=self.sequence_length,
                )
                self.vocab = temp_dataset.vocab

            self.test_dataset = TokenizedTextDataset(
                text_data=test_text,
                tokenizer=self.tokenizer,
                vocab=self.vocab,
                sequence_length=self.sequence_length,
                stride=self.stride,
            )


class WikiText103DataModule(WikiText2DataModule):
    """
    WikiText-103 dataset for language modeling.

    A larger version of WikiText with ~100M tokens.
    """

    def prepare_data(self) -> None:
        """Download WikiText-103 data."""
        if not TORCHTEXT_AVAILABLE:
            raise ImportError("torchtext is required for WikiText datasets")

        WikiText103(self.data_dir, split="train")
        WikiText103(self.data_dir, split="valid")
        WikiText103(self.data_dir, split="test")

    def setup(self, stage: str | None = None) -> None:
        """Setup WikiText-103 datasets."""
        if not TORCHTEXT_AVAILABLE:
            raise ImportError("torchtext is required for WikiText datasets")

        if stage == "fit" or stage is None:
            train_iter = WikiText103(self.data_dir, split="train")
            train_text = " ".join(train_iter)

            self.train_dataset = TokenizedTextDataset(
                text_data=train_text,
                tokenizer=self.tokenizer,
                vocab=None,
                max_vocab_size=self.max_vocab_size,
                sequence_length=self.sequence_length,
                stride=self.stride,
            )

            self.vocab = self.train_dataset.vocab

            val_iter = WikiText103(self.data_dir, split="valid")
            val_text = " ".join(val_iter)

            self.val_dataset = TokenizedTextDataset(
                text_data=val_text,
                tokenizer=self.tokenizer,
                vocab=self.vocab,
                sequence_length=self.sequence_length,
                stride=self.stride,
            )

        if stage == "test" or stage is None:
            test_iter = WikiText103(self.data_dir, split="test")
            test_text = " ".join(test_iter)

            if self.vocab is None:
                train_iter = WikiText103(self.data_dir, split="train")
                train_text = " ".join(train_iter)
                temp_dataset = TokenizedTextDataset(
                    text_data=train_text,
                    tokenizer=self.tokenizer,
                    max_vocab_size=self.max_vocab_size,
                    sequence_length=self.sequence_length,
                )
                self.vocab = temp_dataset.vocab

            self.test_dataset = TokenizedTextDataset(
                text_data=test_text,
                tokenizer=self.tokenizer,
                vocab=self.vocab,
                sequence_length=self.sequence_length,
                stride=self.stride,
            )


class CustomTextDataModule(BaseTextDataModule):
    """
    Custom text DataModule for arbitrary text data.

    Load text from files or strings.
    """

    def __init__(
        self,
        text_data: str | list[str] | None = None,
        text_file: str | None = None,
        data_dir: str | None = None,
        tokenizer_type: Literal["char", "word", "basic"] = "char",
        max_vocab_size: int = 10000,
        sequence_length: int = 128,
        stride: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize custom text DataModule.

        Args:
            text_data: Raw text string or list of strings
            text_file: Path to text file (alternative to text_data)
            data_dir: Data directory
            tokenizer_type: Type of tokenizer
            max_vocab_size: Maximum vocabulary size
            sequence_length: Sequence length
            stride: Stride for sliding window
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Pin memory
            val_split: Validation split fraction
            test_split: Test split fraction
            seed: Random seed
        """
        super().__init__(
            data_dir=data_dir,
            tokenizer_type=tokenizer_type,
            max_vocab_size=max_vocab_size,
            sequence_length=sequence_length,
            stride=stride,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            seed=seed,
        )

        self.text_data = text_data
        self.text_file = text_file
        self.test_split = test_split

        if text_data is None and text_file is None:
            raise ValueError("Must provide either text_data or text_file")

    def prepare_data(self) -> None:
        """Load text data."""
        if self.text_file is not None:
            # Load from file
            with open(self.text_file, "r", encoding="utf-8") as f:
                self.text_data = f.read()

    def setup(self, stage: str | None = None) -> None:
        """Setup custom text datasets."""
        # Ensure data is loaded
        if self.text_data is None:
            self.prepare_data()

        # Create full dataset
        full_dataset = TokenizedTextDataset(
            text_data=self.text_data,
            tokenizer=self.tokenizer,
            vocab=None,
            max_vocab_size=self.max_vocab_size,
            sequence_length=self.sequence_length,
            stride=self.stride,
        )

        self.vocab = full_dataset.vocab

        # Split into train/val/test
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == "test" or stage is None:
            self.test_dataset = test_dataset
