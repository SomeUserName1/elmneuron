"""
Routing transforms for mapping inputs to synapses in ELM models.

Routing strategies determine how input features are assigned to
synapses, which is particularly important for Branch-ELM where
multiple synapses connect to each branch.
"""

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn


class RoutingTransform(ABC, nn.Module):
    """
    Abstract base class for routing transforms.

    Routing transforms map input features to synapses, potentially
    changing the dimensionality and/or ordering of the input.
    """

    def __init__(self, num_input: int, num_synapse: int):
        """
        Initialize routing transform.

        Args:
            num_input: Number of input features
            num_synapse: Number of synapses (output dimension)
        """
        super().__init__()
        self.num_input = num_input
        self.num_synapse = num_synapse

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply routing transform to input.

        Args:
            x: Input tensor (..., num_input)

        Returns:
            Routed tensor (..., num_synapse)
        """
        pass

    def get_routing_info(self) -> dict:
        """
        Get information about the routing configuration.

        Returns:
            Dictionary with routing metadata
        """
        return {
            "type": self.__class__.__name__,
            "num_input": self.num_input,
            "num_synapse": self.num_synapse,
        }


class IdentityRouting(RoutingTransform):
    """
    Identity routing (no transformation).

    Requires num_input == num_synapse.
    """

    def __init__(self, num_input: int):
        """
        Initialize identity routing.

        Args:
            num_input: Number of input features (= num_synapse)
        """
        super().__init__(num_input, num_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply identity routing (no-op).

        Args:
            x: Input tensor (..., num_input)

        Returns:
            Same tensor (..., num_input)
        """
        return x


class RandomRouting(RoutingTransform):
    """
    Random routing that samples input indices for each synapse.

    Each synapse randomly selects one input feature.
    Useful for oversampling (num_synapse > num_input).
    """

    def __init__(
        self,
        num_input: int,
        num_synapse: int,
        seed: int | None = None,
    ):
        """
        Initialize random routing.

        Args:
            num_input: Number of input features
            num_synapse: Number of synapses
            seed: Random seed for reproducibility (default: None)
        """
        super().__init__(num_input, num_synapse)

        # Generate random routing indices
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randint(0, num_input, (num_synapse,), generator=generator)
        else:
            indices = torch.randint(0, num_input, (num_synapse,))

        # Register as buffer (non-trainable, but saved with model)
        self.register_buffer("routing_indices", indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random routing.

        Args:
            x: Input tensor (..., num_input)

        Returns:
            Routed tensor (..., num_synapse)
        """
        # Use index_select on last dimension
        return torch.index_select(x, -1, self.routing_indices)

    def get_routing_info(self) -> dict:
        """Get routing information including indices."""
        info = super().get_routing_info()
        info["routing_indices"] = self.routing_indices.tolist()
        return info


class NeuronIORouting(RoutingTransform):
    """
    NeuronIO-specific routing for biophysical neuron modeling.

    This routing strategy:
    1. Interleaves excitatory and inhibitory inputs
    2. Creates overlapping windows that assign neighboring
       inputs to the same branch (exploits spatial locality)

    This is the routing strategy used in the original NeuronIO
    experiments and is crucial for Branch-ELM performance on
    biophysical neuron data.
    """

    def __init__(
        self,
        num_input: int,
        num_branch: int,
        num_synapse_per_branch: int,
    ):
        """
        Initialize NeuronIO routing.

        Args:
            num_input: Number of input features
            num_branch: Number of branches
            num_synapse_per_branch: Number of synapses per branch

        Raises:
            AssertionError: If configuration is invalid
        """
        num_synapse = num_branch * num_synapse_per_branch
        super().__init__(num_input, num_synapse)

        self.num_branch = num_branch
        self.num_synapse_per_branch = num_synapse_per_branch

        # Validate configuration
        assert (
            math.ceil(num_input / num_branch) <= num_synapse_per_branch
        ), f"Insufficient synapses per branch: need at least {math.ceil(num_input / num_branch)}, got {num_synapse_per_branch}"

        # Create interlocking indices (excitatory/inhibitory interleaving)
        interlocking_indices = self._create_interlocking_indices(num_input)

        # Create overlapping window indices (spatial locality)
        overlapping_indices, valid_mask = self._create_overlapping_window_indices(
            num_input, num_branch, num_synapse_per_branch
        )

        # Compose the two transformations
        routing_indices = interlocking_indices[overlapping_indices]

        # Register as buffers
        self.register_buffer("routing_indices", routing_indices)
        self.register_buffer("valid_mask", valid_mask.float())

    @staticmethod
    def _create_interlocking_indices(num_input: int) -> torch.Tensor:
        """
        Create interlocking indices for excitatory/inhibitory interleaving.

        This assumes the first half of inputs are excitatory and the
        second half are inhibitory, then interleaves them.

        Args:
            num_input: Number of input features

        Returns:
            Interlocking index mapping
        """
        half_num_input = num_input // 2
        half_range_steps = (torch.arange(num_input) % 2) * half_num_input
        single_steps = torch.div(torch.arange(num_input), 2, rounding_mode="floor")
        return half_range_steps + single_steps

    @staticmethod
    def _create_overlapping_window_indices(
        num_input: int, num_windows: int, num_elements_per_window: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create overlapping window indices for spatial locality.

        Each window (branch) receives a contiguous set of inputs
        with overlap between adjacent windows.

        Args:
            num_input: Number of input features
            num_windows: Number of windows (branches)
            num_elements_per_window: Elements per window (synapses per branch)

        Returns:
            tuple of (indices, valid_mask):
            - indices: Flat tensor of window indices
            - valid_mask: Binary mask for valid indices
        """
        stride_size = math.ceil(num_input / num_windows)

        # Create windows
        overlapping_indices = (
            torch.arange(num_windows).unsqueeze(1) * stride_size
        ) + torch.arange(num_elements_per_window).unsqueeze(0)

        # Mark valid indices
        valid_indices = overlapping_indices < num_input

        # Clamp to valid range
        overlapping_indices = torch.clamp(overlapping_indices, max=num_input - 1)

        return overlapping_indices.flatten(), valid_indices.flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply NeuronIO routing.

        Args:
            x: Input tensor (..., num_input)

        Returns:
            Routed tensor (..., num_synapse) with masking applied
        """
        # Apply routing
        routed = torch.index_select(x, -1, self.routing_indices)

        # Apply validity mask
        routed = routed * self.valid_mask

        return routed

    def get_routing_info(self) -> dict:
        """Get routing information."""
        info = super().get_routing_info()
        info.update(
            {
                "num_branch": self.num_branch,
                "num_synapse_per_branch": self.num_synapse_per_branch,
                "num_valid_synapses": int(self.valid_mask.sum().item()),
            }
        )
        return info
