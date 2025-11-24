"""
Expressive Leaky Memory (ELM) Neuron

A biologically-inspired phenomenological model of cortical neurons that efficiently
captures sophisticated neuronal computations.

This package provides two variants:
- ELM v1: Original implementation with ~53K parameters for NeuronIO
- Branch-ELM v2: Improved variant with ~8K parameters (7x reduction)

Example usage:
    >>> from elmneuron import ELM_v2
    >>> model = ELM_v2(
    ...     num_input=1278,
    ...     num_output=2,
    ...     num_memory=15,
    ...     lambda_value=5.0,
    ...     num_branch=45,
    ...     num_synapse_per_branch=100,
    ... )

For more information, see the README.md and CLAUDE.md files.
"""

__version__ = "0.1.0"
__author__ = "Aaron Spieler"
__license__ = "MIT"

# Import main model classes
# Note: ELM v1 has been deprecated, only v2 is available
from elmneuron.expressive_leaky_memory_neuron_v2 import ELM as ELM_v2

# For backwards compatibility
ELM = ELM_v2

# Import utility functions
from elmneuron.modeling_utils import (
    MLP,
    create_interlocking_indices,
    create_overlapping_window_indices,
    custom_tanh,
    inverse_scaled_sigmoid,
    scaled_sigmoid,
)

__all__ = [
    # Version
    "__version__",
    # Main models
    "ELM",
    "ELM_v2",
    # Utilities
    "MLP",
    "create_interlocking_indices",
    "create_overlapping_window_indices",
    "custom_tanh",
    "inverse_scaled_sigmoid",
    "scaled_sigmoid",
]
