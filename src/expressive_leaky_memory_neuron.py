"""Expressive Leaky Memory (ELM) Neuron - Version 1.

This module implements the original ELM neuron architecture, a biologically-inspired
recurrent neural network unit designed for processing temporal spike-based data.

The ELM neuron features:
    - Fast synaptic dynamics for input integration
    - Intermediate branch aggregation layer
    - Slow memory dynamics with heterogeneous time constants
    - Multi-timescale processing for capturing both transient and persistent patterns

Key differences from standard RNNs:
    - Explicit time constants (interpretable temporal scales)
    - Multi-timescale dynamics built into architecture
    - Biological inspiration from synapses, dendrites, and somatic integration
    - Leaky integration instead of gating mechanisms

Typical usage:
    >>> model = ELM(num_input=128, num_output=2, num_memory=100)
    >>> outputs = model(inputs)  # inputs: (batch, time, features)

For more details on the architecture and mathematical formulation, see the
class docstring for `ELM`.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import jit

from .modeling_utils import (
    MLP,
    create_interlocking_indices,
    create_overlapping_window_indices,
    custom_tanh,
    inverse_scaled_sigmoid,
    scaled_sigmoid,
)
from .neuronio.neuronio_data_utils import DEFAULT_Y_TRAIN_SOMA_SCALE

# Supported input routing strategies
PREPROCESS_CONFIGURATIONS = [None, "random_routing", "neuronio_routing"]


class ELM(jit.ScriptModule):
    """Expressive Leaky Memory (ELM) neuron model (v1).

    A biologically-inspired recurrent neural network unit that combines fast synaptic
    dynamics with slower memory dynamics to process temporal spike-based data.

    Architecture:
        1. Synapse Layer: Fast-decaying states that receive and filter input signals
        2. Branch Layer: Intermediate aggregation of synaptic inputs into branches
        3. Memory Layer: Slow-decaying states that maintain long-term dependencies

    Dynamics:
        At each timestep t, the model computes:

        Synapse update:
            s_t = κ_s * s_{t-1} + w_s * x_t

        Branch aggregation:
            b_t = Σ_{j∈branch_i} s_t[j]

        Memory update:
            Δm_t = tanh(MLP([b_t; κ_m * m_{t-1}]))
            m_t = κ_m * m_{t-1} + λ * (1 - κ_m) * Δm_t

        Output:
            y_t = W_y * m_t

    where:
        - κ_s, κ_m are decay factors derived from time constants τ_s, τ_m
        - w_s are learnable synaptic weights
        - λ is a scaling factor for memory updates
        - MLP is a multi-layer perceptron

    Args:
        num_input: Number of input features/channels
        num_output: Number of output features/channels
        num_memory: Number of memory units (slow dynamics). Default: 100
        lambda_value: Memory update scaling factor (λ). Default: 5.0
        mlp_num_layers: Number of hidden layers in the MLP. Default: 1
        mlp_hidden_size: Hidden dimension of MLP layers. Default: 2*num_memory
        mlp_activation: Activation function ("relu" or "silu"). Default: "relu"
        tau_s_value: Synapse time constant in timesteps. Default: 5.0
        memory_tau_min: Minimum memory time constant in timesteps. Default: 1.0
        memory_tau_max: Maximum memory time constant in timesteps. Default: 1000.0
        learn_memory_tau: Whether memory time constants are learnable. Default: False
        w_s_value: Initial value for synaptic weights. Default: 0.5
        num_branch: Number of branches (groups of synapses). Default: num_input
        num_synapse_per_branch: Number of synapses per branch. Default: 1
        input_to_synapse_routing: Input routing strategy (None, "random_routing",
            "neuronio_routing"). Default: None
        delta_t: Temporal resolution (timestep size). Default: 1.0

    Example:
        >>> model = ELM(num_input=128, num_output=2, num_memory=100)
        >>> inputs = torch.randn(32, 1000, 128)  # (batch, time, features)
        >>> outputs = model(inputs)  # (32, 1000, 2)
    """

    __constants__ = [
        "num_input",
        "num_output",
        "num_memory",
        "lambda_value",
        "mlp_num_layers",
        "mlp_activation",
        "memory_tau_min",
        "memory_tau_max",
        "learn_memory_tau",
        "w_s_value",
        "num_synapse_per_branch",
        "input_to_synapse_routing",
        "delta_t",
    ]

    def __init__(
        self,
        num_input: int,
        num_output: int,
        num_memory: int = 100,
        lambda_value: float = 5.0,
        mlp_num_layers: int = 1,
        mlp_hidden_size: Optional[int] = None,
        mlp_activation: str = "relu",
        tau_s_value: float = 5.0,
        memory_tau_min: float = 1.0,
        memory_tau_max: float = 1000.0,
        learn_memory_tau: bool = False,
        w_s_value: float = 0.5,
        num_branch: Optional[int] = None,
        num_synapse_per_branch: int = 1,
        input_to_synapse_routing: Optional[str] = None,
        delta_t: float = 1.0,
    ):
        super(ELM, self).__init__()

        # Store basic configuration
        self.num_input, self.num_output = num_input, num_output
        self.num_memory = num_memory
        self.lambda_value = lambda_value
        self.mlp_num_layers = mlp_num_layers
        self.mlp_activation = mlp_activation
        self.memory_tau_min, self.memory_tau_max = memory_tau_min, memory_tau_max
        self.learn_memory_tau = learn_memory_tau
        self.tau_s_value, self.w_s_value = tau_s_value, w_s_value
        self.num_synapse_per_branch = num_synapse_per_branch
        self.input_to_synapse_routing = input_to_synapse_routing
        self.delta_t = delta_t

        # Derive neuron architecture properties
        # Hidden size defaults to 2x memory dimension if not specified
        self.mlp_hidden_size = mlp_hidden_size if mlp_hidden_size else 2 * num_memory
        # Number of branches defaults to number of inputs (1-to-1 mapping)
        self.num_branch = self.num_input if num_branch is None else num_branch
        # MLP receives branch activities concatenated with memory states
        self.num_mlp_input = self.num_branch + num_memory
        # Total synapses = synapses per branch × number of branches
        self.num_synapse = num_synapse_per_branch * self.num_branch

        # Validate configuration
        # Either num_synapse matches num_input (direct mapping) or routing is specified
        assert self.num_synapse == num_input or input_to_synapse_routing is not None
        assert self.input_to_synapse_routing in PREPROCESS_CONFIGURATIONS

        # Initialize learnable components
        # MLP: maps [branch_states; memory_states] → memory_updates
        self.mlp = MLP(
            self.num_mlp_input,
            self.mlp_hidden_size,
            num_memory,
            mlp_num_layers,
            mlp_activation,
        )
        # Synaptic weights: will be constrained to non-negative via ReLU
        self._proto_w_s = nn.parameter.Parameter(
            torch.full((self.num_synapse,), w_s_value)
        )
        # Output projection: maps memory states to output
        self.w_y = nn.Linear(num_memory, num_output)

        # Initialize synapse time constants (fixed, not learned)
        # All synapses share the same time constant in v1
        tau_s = torch.full((self.num_synapse,), tau_s_value)
        self.tau_s = nn.parameter.Parameter(tau_s, requires_grad=False)

        # Initialize memory time constants (optionally learnable)
        # Log-spaced distribution across [memory_tau_min, memory_tau_max]
        # This creates heterogeneous timescales for multi-scale temporal processing
        _proto_tau_m = torch.logspace(
            math.log10(memory_tau_min + 1e-6),
            math.log10(memory_tau_max - 1e-6),
            num_memory,
        )
        # Transform to unconstrained space for optimization
        # During forward pass, inverse transform ensures values stay in [min, max]
        _proto_tau_m = inverse_scaled_sigmoid(
            _proto_tau_m, memory_tau_min, memory_tau_max
        )
        self._proto_tau_m = nn.parameter.Parameter(
            _proto_tau_m, requires_grad=learn_memory_tau
        )

        # Create input-to-synapse routing indices
        # Stored as parameters for serialization and device transfer
        routing_artifacts = self.create_input_to_synapse_indices()
        self.input_to_synapse_indices = nn.parameter.Parameter(
            routing_artifacts[0], requires_grad=False
        )
        # Mask for handling padding in overlapping routing schemes
        self.valid_indices_mask = nn.parameter.Parameter(
            routing_artifacts[1], requires_grad=False
        )

    @property
    def tau_m(self):
        """Memory time constants in timesteps.

        Returns:
            torch.Tensor: Shape (num_memory,), values in [memory_tau_min, memory_tau_max].
                Longer time constants → slower decay → longer memory.
        """
        return scaled_sigmoid(
            self._proto_tau_m, self.memory_tau_min, self.memory_tau_max
        )

    @property
    def kappa_m(self):
        """Memory decay factors.

        Returns:
            torch.Tensor: Shape (num_memory,), values in (0, 1).
                Computed as κ_m = exp(-Δt / τ_m).
                Higher values → slower decay → more persistent memory.
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))

    @property
    def kappa_s(self):
        """Synapse decay factors.

        Returns:
            torch.Tensor: Shape (num_synapse,), values in (0, 1).
                Computed as κ_s = exp(-Δt / τ_s).
                Higher values → slower decay → longer synaptic integration.
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_s, min=1e-6))

    @property
    def w_s(self):
        """Synaptic weights (non-negative).

        Returns:
            torch.Tensor: Shape (num_synapse,), values >= 0.
                Learnable weights constrained to non-negative via ReLU activation.
        """
        return torch.relu(self._proto_w_s)

    def create_input_to_synapse_indices(self):
        """Create indices for routing inputs to synapses.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                - input_to_synapse_indices: Indices for selecting inputs, shape (num_synapse,)
                - valid_indices_mask: Binary mask for valid connections, shape (num_synapse,)
                Returns (None, None) if no routing is specified.

        Routing strategies:
            - "random_routing": Random assignment of inputs to synapses
            - "neuronio_routing": Specialized routing for biological data:
                1. Interlaces excitatory/inhibitory inputs
                2. Creates overlapping windows for spatial locality
                3. Assigns neighboring inputs to the same branch
        """
        if self.input_to_synapse_routing == "random_routing":
            # Randomly select num_synapse inputs from num_input channels
            # This creates a sparse, random connectivity pattern
            input_to_synapse_indices = torch.randint(
                self.num_input, (self.num_synapse,)
            )
            # All connections are valid (no masking needed)
            return input_to_synapse_indices, torch.ones_like(input_to_synapse_indices)

        elif self.input_to_synapse_routing == "neuronio_routing":
            # Validate that windows can accommodate all inputs
            assert (
                math.ceil(self.num_input / self.num_branch)
                <= self.num_synapse_per_branch
            ), "num_synapse_per_branch too small for neuronio_routing"

            # Step 1: Interlace excitatory (even) and inhibitory (odd) inputs
            # Pattern: [0, N/2, 1, N/2+1, 2, N/2+2, ...]
            interlocking_indices = create_interlocking_indices(self.num_input)

            # Step 2: Create overlapping windows for spatial receptive fields
            # Each branch receives inputs from a sliding window
            overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
                self.num_input, self.num_branch, self.num_synapse_per_branch
            )

            # Combine: apply interlocking to windowed indices
            input_to_synapse_indices = interlocking_indices[overlapping_indices]

            return input_to_synapse_indices, valid_indices_mask
        else:
            # No routing: assume direct 1-to-1 mapping
            return None, None

    def route_input_to_synapses(self, x):
        """Apply input-to-synapse routing transformation.

        Args:
            x: Input tensor, shape (batch_size, T, num_input)

        Returns:
            torch.Tensor: Routed inputs, shape (batch_size, T, num_synapse)
        """
        if self.input_to_synapse_routing is not None:
            # Select specified input channels for each synapse
            x = torch.index_select(x, 2, self.input_to_synapse_indices)
            # Apply validity mask (zeros out padded connections)
            x = x * self.valid_indices_mask
        return x

    @jit.script_method
    def dynamics(self, x, s_prev, m_prev, w_s, kappa_s, kappa_m):
        """Compute the ELM dynamics for a single timestep.

        This is the core recurrent update that implements the three-stage
        computation: synapse → branch → memory → output.

        Args:
            x: Input at time t, shape (batch_size, num_synapse)
            s_prev: Previous synapse states, shape (batch_size, num_synapse)
            m_prev: Previous memory states, shape (batch_size, num_memory)
            w_s: Synaptic weights, shape (num_synapse,)
            kappa_s: Synapse decay factors, shape (num_synapse,)
            kappa_m: Memory decay factors, shape (num_memory,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - y_t: Output at time t, shape (batch_size, num_output)
                - s_t: Updated synapse states, shape (batch_size, num_synapse)
                - m_t: Updated memory states, shape (batch_size, num_memory)
        """
        batch_size, _ = x.shape

        # Synapse dynamics: leaky integration of weighted inputs
        # s_t = κ_s * s_{t-1} + w_s * x_t
        s_t = kappa_s * s_prev + w_s * x

        # Branch aggregation: sum synapses within each branch
        # Reshape: (batch, num_synapse) → (batch, num_branch, num_synapse_per_branch)
        # Then sum over synapses per branch
        syn_input = s_t.view(batch_size, self.num_branch, -1).sum(dim=-1)

        # Memory update: MLP processes [branch_states; decayed_memory]
        # Δm_t = tanh(MLP([b_t; κ_m * m_{t-1}]))
        delta_m_t = custom_tanh(
            self.mlp(torch.cat([syn_input, kappa_m * m_prev], dim=-1))
        )

        # Memory dynamics: leaky integration with scaled update
        # m_t = κ_m * m_{t-1} + λ * (1 - κ_m) * Δm_t
        # The (1 - κ_m) term ensures consistent update magnitude across timescales
        m_t = kappa_m * m_prev + self.lambda_value * (1 - kappa_m) * delta_m_t

        # Output projection: linear mapping from memory to output space
        y_t = self.w_y(m_t)

        return y_t, s_t, m_t

    @jit.script_method
    def forward(self, X):
        """Forward pass through the ELM neuron.

        Processes a batch of input sequences by iteratively applying the
        dynamics function for each timestep. States are initialized to zero
        at the start of each sequence.

        Args:
            X: Input sequences, shape (batch_size, T, num_input)
                - batch_size: Number of sequences
                - T: Sequence length (number of timesteps)
                - num_input: Input feature dimension

        Returns:
            torch.Tensor: Output sequences, shape (batch_size, T, num_output)
        """
        batch_size, T, _ = X.shape

        # Get current parameter values
        w_s = self.w_s
        kappa_s, kappa_m = self.kappa_s, self.kappa_m

        # Initialize synapse and memory states to zero
        s_prev = torch.zeros(batch_size, len(kappa_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Collect outputs for each timestep
        outputs = torch.jit.annotate(List[torch.Tensor], [])

        # Apply input routing (if configured)
        inputs = self.route_input_to_synapses(X)

        # Iterate through sequence, updating states at each timestep
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)

        # Stack outputs: list of (batch, num_output) → (batch, T, num_output)
        return torch.stack(outputs, dim=-2)

    @jit.script_method
    def neuronio_eval_forward(
        self, X, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        """Forward pass for NeuronIO evaluation with post-processing.

        Specialized forward pass for the NeuronIO dataset that applies
        appropriate activations and scaling to raw model outputs.

        Args:
            X: Input sequences, shape (batch_size, T, num_input)
            y_train_soma_scale: Scaling factor used during training for soma voltage.
                Default: DEFAULT_Y_TRAIN_SOMA_SCALE

        Returns:
            torch.Tensor: Predictions, shape (batch_size, T, 2) where:
                - [..., 0]: Spike probabilities in [0, 1] (sigmoid-activated)
                - [..., 1]: Soma voltage predictions (rescaled to original units)
        """
        # Get raw outputs from model
        outputs = self.forward(X)
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]

        # Apply sigmoid to spike prediction → probability in [0, 1]
        spike_pred = torch.sigmoid(spike_pred)

        # Rescale soma prediction back to original voltage units
        soma_pred = 1 / y_train_soma_scale * soma_pred

        return torch.stack([spike_pred, soma_pred], dim=-1)

    @jit.script_method
    def neuronio_viz_forward(
        self, X, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        """Forward pass with internal state recording for visualization.

        Similar to neuronio_eval_forward but also returns the complete history
        of internal synapse and memory states for analysis and visualization.

        Args:
            X: Input sequences, shape (batch_size, T, num_input)
            y_train_soma_scale: Scaling factor used during training for soma voltage.
                Default: DEFAULT_Y_TRAIN_SOMA_SCALE

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - outputs: Predictions, shape (batch_size, T, 2)
                    - [..., 0]: Spike probabilities
                    - [..., 1]: Soma voltage predictions
                - s_record: Synapse state history, shape (batch_size, T, num_synapse)
                - m_record: Memory state history, shape (batch_size, T, num_memory)

        Note:
            This method is more memory-intensive than neuronio_eval_forward due to
            storing all internal states. Use only when visualization is needed.
        """
        batch_size, T, _ = X.shape

        # Get current parameter values
        w_s = self.w_s
        kappa_s, kappa_m = self.kappa_s, self.kappa_m

        # Initialize states to zero
        s_prev = torch.zeros(batch_size, len(kappa_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Prepare storage for outputs and internal states
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        s_record = torch.jit.annotate(List[torch.Tensor], [])
        m_record = torch.jit.annotate(List[torch.Tensor], [])

        # Apply input routing
        inputs = self.route_input_to_synapses(X)

        # Iterate through sequence, recording states at each timestep
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)
            s_record.append(s_prev)  # Record synapse states
            m_record.append(m_prev)  # Record memory states

        # Stack all recorded values
        outputs = torch.stack(outputs, dim=-2)
        s_record = torch.stack(s_record, dim=-2)
        m_record = torch.stack(m_record, dim=-2)

        # Post-process outputs (same as neuronio_eval_forward)
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = 1 / y_train_soma_scale * soma_pred
        outputs = torch.stack([spike_pred, soma_pred], dim=-1)

        return outputs, s_record, m_record
