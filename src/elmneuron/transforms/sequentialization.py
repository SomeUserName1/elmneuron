"""
Sequentialization strategies for converting non-sequential data to sequences.

These transforms enable ELM models to process images, text, and other
non-sequential data by converting them into temporal sequences.
"""

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceTransform(ABC, nn.Module):
    """
    Abstract base class for sequentialization transforms.

    Sequentialization transforms convert non-sequential data (like images)
    into sequences that can be processed by temporal models like ELM.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input to sequence.

        Args:
            x: Input tensor

        Returns:
            Sequence tensor (batch, time, features)
        """
        pass

    def get_sequence_info(self) -> dict:
        """
        Get information about the sequentialization.

        Returns:
            Dictionary with sequence metadata
        """
        return {"type": self.__class__.__name__}


class PatchSequence(SequenceTransform):
    """
    Convert images to sequences of patches.

    Splits an image into non-overlapping patches and arranges them
    as a sequence. Useful for Vision Transformer-style processing.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        order: Literal["raster", "random", "spiral"] = "raster",
        flatten: bool = True,
        seed: int | None = None,
    ):
        """
        Initialize patch sequence transform.

        Args:
            patch_size: Size of patches (int or (height, width))
            order: Order of patches in sequence:
                - 'raster': row-major order (left-to-right, top-to-bottom)
                - 'random': random permutation
                - 'spiral': spiral from center outward
            flatten: Whether to flatten patches (default: True)
            seed: Random seed for 'random' order (default: None)
        """
        super().__init__()

        if isinstance(patch_size, int):
            self.patch_h = self.patch_w = patch_size
        else:
            self.patch_h, self.patch_w = patch_size

        self.order = order
        self.flatten = flatten
        self.seed = seed
        self._permutation = None  # Computed lazily

    def _get_patch_permutation(
        self, num_patches_h: int, num_patches_w: int
    ) -> torch.Tensor:
        """Compute patch ordering permutation."""
        if self._permutation is not None:
            return self._permutation

        num_patches = num_patches_h * num_patches_w

        if self.order == "raster":
            # Row-major order (default)
            perm = torch.arange(num_patches)

        elif self.order == "random":
            # Random permutation
            generator = torch.Generator()
            if self.seed is not None:
                generator.manual_seed(self.seed)
            perm = torch.randperm(num_patches, generator=generator)

        elif self.order == "spiral":
            # Spiral from center (simple approximation)
            # Create grid coordinates
            i, j = torch.meshgrid(
                torch.arange(num_patches_h), torch.arange(num_patches_w), indexing="ij"
            )
            # Compute distance from center
            center_i, center_j = num_patches_h / 2, num_patches_w / 2
            dist = (i - center_i) ** 2 + (j - center_j) ** 2
            # Sort by distance
            perm = torch.argsort(dist.flatten())

        else:
            raise ValueError(f"Unknown order: {self.order}")

        self._permutation = perm
        return perm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image(s) to patch sequence.

        Args:
            x: Image tensor (batch, channels, height, width)

        Returns:
            Patch sequence (batch, num_patches, patch_features)
        """
        batch_size, channels, height, width = x.shape

        # Check divisibility
        assert (
            height % self.patch_h == 0
        ), f"Height {height} not divisible by patch height {self.patch_h}"
        assert (
            width % self.patch_w == 0
        ), f"Width {width} not divisible by patch width {self.patch_w}"

        num_patches_h = height // self.patch_h
        num_patches_w = width // self.patch_w

        # Extract patches using unfold
        # Shape: (batch, channels, num_patches_h, num_patches_w, patch_h, patch_w)
        patches = x.unfold(2, self.patch_h, self.patch_h).unfold(
            3, self.patch_w, self.patch_w
        )

        # Reshape to (batch, num_patches_h * num_patches_w, channels, patch_h, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(
            batch_size,
            num_patches_h * num_patches_w,
            channels,
            self.patch_h,
            self.patch_w,
        )

        # Apply permutation if needed
        if self.order != "raster":
            perm = self._get_patch_permutation(num_patches_h, num_patches_w).to(
                x.device
            )
            patches = patches[:, perm]

        # Flatten patches if requested
        if self.flatten:
            patches = patches.flatten(
                2
            )  # (batch, num_patches, channels * patch_h * patch_w)

        return patches

    def get_sequence_info(self) -> dict:
        """Get sequentialization information."""
        info = super().get_sequence_info()
        info.update(
            {
                "patch_size": (self.patch_h, self.patch_w),
                "order": self.order,
                "flatten": self.flatten,
            }
        )
        return info


class PixelSequence(SequenceTransform):
    """
    Convert images to sequences of pixels.

    Flattens an image into a sequence of pixels, optionally
    with different scanning orders.
    """

    def __init__(
        self,
        order: Literal["raster", "snake", "hilbert"] = "raster",
        flatten_channels: bool = True,
    ):
        """
        Initialize pixel sequence transform.

        Args:
            order: Scanning order:
                - 'raster': row-major (left-to-right, top-to-bottom)
                - 'snake': boustrophedon (alternating direction per row)
                - 'hilbert': Hilbert curve (space-filling curve)
            flatten_channels: Whether to flatten color channels (default: True)
        """
        super().__init__()
        self.order = order
        self.flatten_channels = flatten_channels
        self._permutation = None

    def _get_hilbert_curve(self, n: int) -> torch.Tensor:
        """
        Generate Hilbert curve indices for n×n grid.

        Note: This is a simplified version for powers of 2.
        """

        def hilbert_d2xy(n, d):
            """Convert Hilbert curve distance to (x, y)."""
            x = y = 0
            s = 1
            while s < n:
                rx = 1 & (d // 2)
                ry = 1 & (d ^ rx)
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                d //= 4
                s *= 2
            return x, y

        indices = []
        for d in range(n * n):
            x, y = hilbert_d2xy(n, d)
            indices.append(y * n + x)

        return torch.tensor(indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image(s) to pixel sequence.

        Args:
            x: Image tensor (batch, channels, height, width)

        Returns:
            Pixel sequence (batch, height * width, features)
        """
        batch_size, channels, height, width = x.shape

        if self.order == "raster":
            # Simple flatten
            if self.flatten_channels:
                # (batch, channels, height, width) -> (batch, height * width, channels)
                sequence = x.permute(0, 2, 3, 1).contiguous()
                sequence = sequence.view(batch_size, height * width, channels)
            else:
                # (batch, channels, height, width) -> (batch, height * width * channels, 1)
                sequence = x.view(batch_size, channels * height * width, 1)

        elif self.order == "snake":
            # Boustrophedon scanning
            sequence_list = []
            for row in range(height):
                if row % 2 == 0:
                    # Left to right
                    sequence_list.append(x[:, :, row, :])
                else:
                    # Right to left
                    sequence_list.append(x[:, :, row, :].flip(-1))

            sequence = torch.stack(
                sequence_list, dim=2
            )  # (batch, channels, height, width)

            if self.flatten_channels:
                sequence = sequence.permute(0, 2, 3, 1).contiguous()
                sequence = sequence.view(batch_size, height * width, channels)
            else:
                sequence = sequence.view(batch_size, channels * height * width, 1)

        elif self.order == "hilbert":
            # Hilbert curve (only for square images with power-of-2 dimensions)
            assert height == width, "Hilbert curve requires square images"
            assert (
                height & (height - 1)
            ) == 0, "Hilbert curve requires power-of-2 dimensions"

            # Get Hilbert curve permutation
            perm = self._get_hilbert_curve(height).to(x.device)

            # Flatten spatial dimensions
            flat = x.permute(
                0, 2, 3, 1
            ).contiguous()  # (batch, height, width, channels)
            flat = flat.view(batch_size, height * width, channels)

            # Reorder by Hilbert curve
            sequence = flat[:, perm]

        else:
            raise ValueError(f"Unknown order: {self.order}")

        return sequence

    def get_sequence_info(self) -> dict:
        """Get sequentialization information."""
        info = super().get_sequence_info()
        info.update(
            {
                "order": self.order,
                "flatten_channels": self.flatten_channels,
            }
        )
        return info


class ChunkSequence(SequenceTransform):
    """
    Convert data to fixed-size chunks with optional overlap.

    Useful for processing long sequences or flattened data.
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int = 0,
        padding: Literal["zero", "replicate", "circular"] = "zero",
    ):
        """
        Initialize chunk sequence transform.

        Args:
            chunk_size: Size of each chunk
            overlap: Overlap between chunks (default: 0)
            padding: Padding mode if data doesn't divide evenly
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.padding = padding

        assert overlap < chunk_size, "Overlap must be less than chunk_size"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert data to chunk sequence.

        Args:
            x: Input tensor (batch, features) or (batch, time, features)

        Returns:
            Chunk sequence (batch, num_chunks, chunk_size * feature_dim)
        """
        if x.ndim == 2:
            # Add time dimension
            x = x.unsqueeze(1)

        batch_size, seq_len, feature_dim = x.shape

        stride = self.chunk_size - self.overlap

        # Calculate number of chunks needed
        num_chunks = math.ceil((seq_len - self.overlap) / stride)

        # Pad if necessary
        total_len = num_chunks * stride + self.overlap
        if total_len > seq_len:
            pad_len = total_len - seq_len
            if self.padding == "zero":
                padding = torch.zeros(batch_size, pad_len, feature_dim, device=x.device)
            elif self.padding == "replicate":
                padding = x[:, -1:].expand(batch_size, pad_len, feature_dim)
            elif self.padding == "circular":
                padding = x[:, :pad_len]
            else:
                raise ValueError(f"Unknown padding: {self.padding}")

            x = torch.cat([x, padding], dim=1)

        # Extract chunks using unfold
        chunks = x.unfold(
            1, self.chunk_size, stride
        )  # (batch, num_chunks, feature_dim, chunk_size)
        chunks = chunks.permute(
            0, 1, 3, 2
        )  # (batch, num_chunks, chunk_size, feature_dim)
        chunks = chunks.contiguous().view(
            batch_size, num_chunks, self.chunk_size * feature_dim
        )

        return chunks

    def get_sequence_info(self) -> dict:
        """Get sequentialization information."""
        info = super().get_sequence_info()
        info.update(
            {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "padding": self.padding,
            }
        )
        return info


class SlidingWindow(SequenceTransform):
    """
    Create sliding window views of sequential data.

    Useful for temporal data augmentation and context windows.
    """

    def __init__(
        self,
        window_size: int,
        stride: int = 1,
    ):
        """
        Initialize sliding window transform.

        Args:
            window_size: Size of each window
            stride: Stride between windows (default: 1)
        """
        super().__init__()
        self.window_size = window_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sliding windows.

        Args:
            x: Input tensor (batch, time, features)

        Returns:
            Windows (batch, num_windows, window_size, features)
        """
        batch_size, seq_len, feature_dim = x.shape

        # Use unfold to create windows
        windows = x.unfold(
            1, self.window_size, self.stride
        )  # (batch, num_windows, feature_dim, window_size)
        windows = windows.permute(
            0, 1, 3, 2
        )  # (batch, num_windows, window_size, feature_dim)

        return windows

    def get_sequence_info(self) -> dict:
        """Get sequentialization information."""
        info = super().get_sequence_info()
        info.update(
            {
                "window_size": self.window_size,
                "stride": self.stride,
            }
        )
        return info
