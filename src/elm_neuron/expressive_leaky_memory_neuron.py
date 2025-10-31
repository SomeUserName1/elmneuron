"""
Expressive Leaky Memory (ELM) Neuron Model - Version 1.

This module implements the ELM neuron, a biologically-inspired
phenomenological model of cortical neurons. The model efficiently
captures sophisticated neuronal computations using multiple leaky
memory units with learnable timescales and nonlinear synaptic
integration via a Multilayer Perceptron (MLP).

Key architectural components:
1. Synapse dynamics: Filters input through exponential traces
2. Integration mechanism: MLP processes synaptic inputs + memory
3. Memory dynamics: Leaky integrators with multiple timescales
4. Output dynamics: Linear readout from memory state

Reference:
    Spieler, A., Rahaman, N., Martius, G., Schölkopf, B., &
    Levina, A. (2023). The ELM Neuron: an Efficient and Expressive
    Cortical Neuron Model Can Solve Long-Horizon Tasks.
    arXiv preprint arXiv:2306.16922.
"""

import math

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

# Valid preprocessing/routing configurations
PREPROCESS_CONFIGURATIONS = [None, "random_routing", "neuronio_routing"]


class ELM(jit.ScriptModule):
    """
    Expressive Leaky Memory (ELM) neuron model (Version 1).

    The ELM neuron processes inputs through synapse dynamics,
    integrates them via an MLP, and maintains temporal dependencies
    through multiple leaky memory units with distinct timescales.

    **Key differences from Branch-ELM (v2)**:
    - Processes all synaptic inputs directly (no branch reduction)
    - Fixed synapse weights (not learnable)
    - Memory update: m_t = κ_m * m_{t-1} + λ * (1 - κ_m) * Δm_t

    **Architecture flow**:
        x_t → [synapse dynamics] → s_t →
        [MLP(s_t, m_{t-1})] → Δm_t →
        [memory update] → m_t → [linear readout] → y_t

    Args:
        num_input: Input dimension (number of synaptic inputs)
        num_output: Output dimension
        num_memory: Number of memory units (default: 100)
            Typical values: 15-20 for NeuronIO, 100 for long tasks
        lambda_value: Memory update scaling factor (default: 5.0)
            Controls magnitude of memory updates
        mlp_num_layers: Number of MLP hidden layers (default: 1)
        mlp_hidden_size: MLP hidden dimension (default: 2*num_memory)
        mlp_activation: MLP activation function ('relu' or 'silu')
        tau_s_value: Synapse time constant in ms (default: 5.0)
        memory_tau_min: Min memory timescale in ms (default: 1.0)
        memory_tau_max: Max memory timescale in ms (default: 1000.0)
        learn_memory_tau: Whether to learn memory timescales
        w_s_value: Initial synapse weight value (default: 0.5)
        num_branch: Number of branches for routing (default: num_input)
        num_synapse_per_branch: Synapses per branch (default: 1)
        input_to_synapse_routing: Routing strategy
            (None, 'random_routing', 'neuronio_routing')
        delta_t: Fictitious timestep in ms (default: 1.0)

    Attributes:
        tau_m: Memory timescales (property, bounded by tau_min/max)
        kappa_m: Memory decay factors = exp(-delta_t / tau_m)
        kappa_s: Synapse decay factors = exp(-delta_t / tau_s)
        w_s: Synapse weights (property, ReLU-activated)

    Example:
        >>> model = ELM(
        ...     num_input=1278,
        ...     num_output=2,
        ...     num_memory=20,
        ...     lambda_value=5.0
        ... )
        >>> X = torch.randn(8, 100, 1278)  # (batch, time, input)
        >>> Y = model(X)  # (batch, time, output)
    """

    # JIT script constants for compilation
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
        mlp_hidden_size: int | None = None,
        mlp_activation: str = "relu",
        tau_s_value: float = 5.0,
        memory_tau_min: float = 1.0,
        memory_tau_max: float = 1000.0,
        learn_memory_tau: bool = False,
        w_s_value: float = 0.5,
        num_branch: int | None = None,
        num_synapse_per_branch: int = 1,
        input_to_synapse_routing: str | None = None,
        delta_t: float = 1.0,
    ) -> None:
        """Initialize the ELM neuron model."""
        super(ELM, self).__init__()

        # Store basic dimensions
        self.num_input = num_input
        self.num_output = num_output
        self.num_memory = num_memory
        self.lambda_value = lambda_value
        self.mlp_num_layers = mlp_num_layers
        self.mlp_activation = mlp_activation

        # Store timescale bounds
        self.memory_tau_min = memory_tau_min
        self.memory_tau_max = memory_tau_max
        self.learn_memory_tau = learn_memory_tau

        # Store synapse parameters
        self.tau_s_value = tau_s_value
        self.w_s_value = w_s_value
        self.num_synapse_per_branch = num_synapse_per_branch
        self.input_to_synapse_routing = input_to_synapse_routing
        self.delta_t = delta_t

        # Derived properties
        self.mlp_hidden_size = mlp_hidden_size if mlp_hidden_size else 2 * num_memory
        self.num_branch = self.num_input if num_branch is None else num_branch
        self.num_mlp_input = self.num_branch + num_memory
        self.num_synapse = num_synapse_per_branch * self.num_branch

        # Validate configuration
        assert (
            self.num_synapse == num_input or input_to_synapse_routing is not None
        ), "Mismatch: num_synapse != num_input without routing"
        assert (
            self.input_to_synapse_routing in PREPROCESS_CONFIGURATIONS
        ), f"Invalid routing: {input_to_synapse_routing}"

        # Initialize MLP for nonlinear integration
        # Maps [branch_activations, previous_memory] to memory_update
        self.mlp = MLP(
            self.num_mlp_input,
            self.mlp_hidden_size,
            num_memory,
            mlp_num_layers,
            mlp_activation,
        )

        # Initialize synapse weights (fixed, not learned in v1)
        # ReLU activation ensures positivity
        self._proto_w_s = nn.parameter.Parameter(
            torch.full((self.num_synapse,), w_s_value)
        )

        # Initialize output layer (linear readout from memory)
        self.w_y = nn.Linear(num_memory, num_output)

        # Initialize synapse time constants (fixed)
        tau_s = torch.full((self.num_synapse,), tau_s_value)
        self.tau_s = nn.parameter.Parameter(tau_s, requires_grad=False)

        # Initialize memory time constants
        # Logspace distribution ensures diverse timescales
        _proto_tau_m = torch.logspace(
            math.log10(memory_tau_min + 1e-6),
            math.log10(memory_tau_max - 1e-6),
            num_memory,
        )
        # Apply inverse scaled sigmoid for bounded optimization
        _proto_tau_m = inverse_scaled_sigmoid(
            _proto_tau_m, memory_tau_min, memory_tau_max
        )
        self._proto_tau_m = nn.parameter.Parameter(
            _proto_tau_m, requires_grad=learn_memory_tau
        )

        # Create input-to-synapse routing (if specified)
        # Stored as Parameters for JIT compatibility
        routing_artifacts = self.create_input_to_synapse_indices()
        self.input_to_synapse_indices = nn.parameter.Parameter(
            routing_artifacts[0], requires_grad=False
        )
        self.valid_indices_mask = nn.parameter.Parameter(
            routing_artifacts[1], requires_grad=False
        )

    @property
    def tau_m(self) -> torch.Tensor:
        """
        Memory timescales (bounded by tau_min and tau_max).

        Uses scaled sigmoid to ensure timescales remain within
        specified bounds during optimization.

        Returns:
            Tensor of shape (num_memory,) with timescales in ms
        """
        return scaled_sigmoid(
            self._proto_tau_m, self.memory_tau_min, self.memory_tau_max
        )

    @property
    def kappa_m(self) -> torch.Tensor:
        """
        Memory decay factors.

        Computed as exp(-delta_t / tau_m), representing the
        fraction of previous memory retained after one timestep.

        Returns:
            Tensor of shape (num_memory,) with values in (0, 1)
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))

    @property
    def kappa_s(self) -> torch.Tensor:
        """
        Synapse decay factors.

        Computed as exp(-delta_t / tau_s), representing the
        fraction of previous synapse trace retained.

        Returns:
            Tensor of shape (num_synapse,) with values in (0, 1)
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_s, min=1e-6))

    @property
    def w_s(self) -> torch.Tensor:
        """
        Synapse weights (non-negative via ReLU).

        Returns:
            Tensor of shape (num_synapse,) with non-negative values
        """
        return torch.relu(self._proto_w_s)

    def create_input_to_synapse_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create input-to-synapse routing indices and validity mask.

        Implements two routing strategies:
        1. random_routing: Random selection of input indices
        2. neuronio_routing: Structured assignment for
           biophysical neuron modeling (interlocks
           excitatory/inhibitory, creates overlapping windows)

        Returns:
            tuple of (indices, mask):
            - indices: Tensor mapping synapses to input indices
            - mask: Binary mask indicating valid indices
        """
        if self.input_to_synapse_routing == "random_routing":
            # Randomly select num_synapse from num_input
            input_to_synapse_indices = torch.randint(
                self.num_input, (self.num_synapse,)
            )
            return (input_to_synapse_indices, torch.ones_like(input_to_synapse_indices))

        elif self.input_to_synapse_routing == "neuronio_routing":
            # Validate configuration
            assert (
                math.ceil(self.num_input / self.num_branch)
                <= self.num_synapse_per_branch
            ), "Insufficient synapses per branch for input size"

            # Interleave excitatory and inhibitory inputs
            interlocking_indices = create_interlocking_indices(self.num_input)

            # Assign neighboring inputs to same branch
            # (exploits spatial locality)
            overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
                self.num_input, self.num_branch, self.num_synapse_per_branch
            )

            # Compose the two transformations
            input_to_synapse_indices = interlocking_indices[overlapping_indices]

            return input_to_synapse_indices, valid_indices_mask
        else:
            # No routing: direct mapping (None case)
            return (torch.tensor([]), torch.tensor([]))

    def route_input_to_synapses(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply input-to-synapse routing if configured.

        Args:
            x: Input tensor (batch, time, num_input)

        Returns:
            Routed input tensor (batch, time, num_synapse)
        """
        if self.input_to_synapse_routing is not None:
            # Select specified input indices for each synapse
            x = torch.index_select(x, 2, self.input_to_synapse_indices)
            # Apply validity mask (zero out invalid indices)
            x = x * self.valid_indices_mask
        return x

    @jit.script_method
    def dynamics(
        self,
        x: torch.Tensor,
        s_prev: torch.Tensor,
        m_prev: torch.Tensor,
        w_s: torch.Tensor,
        kappa_s: torch.Tensor,
        kappa_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELM dynamics for a single timestep.

        This is the core computation implementing:
        1. Synapse dynamics: s_t = κ_s * s_{t-1} + w_s * x_t
        2. Branch reduction: sum synapses per branch
        3. Memory proposal: Δm_t = tanh(MLP([branch, κ_m*m_{t-1}]))
        4. Memory update: m_t = κ_m*m_{t-1} + λ*(1-κ_m)*Δm_t
        5. Output: y_t = W_y * m_t

        Args:
            x: Current input (batch, num_synapse)
            s_prev: Previous synapse state (batch, num_synapse)
            m_prev: Previous memory state (batch, num_memory)
            w_s: Synapse weights (num_synapse,)
            kappa_s: Synapse decay factors (num_synapse,)
            kappa_m: Memory decay factors (num_memory,)

        Returns:
            tuple of (y_t, s_t, m_t):
            - y_t: Output (batch, num_output)
            - s_t: Updated synapse state (batch, num_synapse)
            - m_t: Updated memory state (batch, num_memory)
        """
        batch_size, _ = x.shape

        # Update synapse traces (exponential filtering)
        # s_t = κ_s * s_{t-1} + w_s * x_t
        s_t = kappa_s * s_prev + w_s * x

        # Reduce synapse traces to branch activations
        # Sum over synapses belonging to each branch
        syn_input = s_t.view(batch_size, self.num_branch, -1).sum(dim=-1)

        # Compute memory update proposal via MLP
        # Input: [branch_activations, decayed_previous_memory]
        # Output: Δm_t (memory update proposal)
        delta_m_t = custom_tanh(
            self.mlp(torch.cat([syn_input, kappa_m * m_prev], dim=-1))
        )

        # Update memory with bounded growth
        # m_t = forget * m_{t-1} + input_scale * update
        # where input_scale = λ * (1 - κ_m) ensures stability
        m_t = kappa_m * m_prev + self.lambda_value * (1 - kappa_m) * delta_m_t

        # Compute output via linear readout
        y_t = self.w_y(m_t)

        return y_t, s_t, m_t

    @jit.script_method
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence through the ELM neuron.

        Iteratively applies dynamics() for each timestep,
        maintaining synapse and memory states across time.

        Args:
            X: Input sequence (batch, time, num_input)

        Returns:
            Output sequence (batch, time, num_output)
        """
        batch_size, T, _ = X.shape

        # Cache frequently used values
        w_s = self.w_s
        kappa_s, kappa_m = self.kappa_s, self.kappa_m

        # Initialize states to zero
        s_prev = torch.zeros(batch_size, len(kappa_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Accumulate outputs (JIT-compatible list annotation)
        outputs = torch.jit.annotate(list[torch.Tensor], [])

        # Apply input routing once for efficiency
        inputs = self.route_input_to_synapses(X)

        # Process each timestep
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)

        # Stack outputs along time dimension
        return torch.stack(outputs, dim=-2)

    @jit.script_method
    def neuronio_eval_forward(
        self, X: torch.Tensor, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ) -> torch.Tensor:
        """
        Forward pass with NeuronIO-specific postprocessing.

        Applies sigmoid to spike predictions and scales soma voltage
        according to NeuronIO dataset conventions.

        Args:
            X: Input sequence (batch, time, num_input)
            y_train_soma_scale: Soma voltage scaling factor

        Returns:
            Postprocessed outputs (batch, time, 2)
            where output[..., 0] = spike probability
            and output[..., 1] = scaled soma voltage
        """
        outputs = self.forward(X)

        # Extract spike and soma predictions
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]

        # Apply sigmoid to spike (probability)
        spike_pred = torch.sigmoid(spike_pred)

        # Rescale soma voltage
        soma_pred = (1 / y_train_soma_scale) * soma_pred

        return torch.stack([spike_pred, soma_pred], dim=-1)

    @jit.script_method
    def neuronio_viz_forward(
        self, X: torch.Tensor, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning outputs and internal states.

        Useful for visualization and analysis of neuron dynamics.

        Args:
            X: Input sequence (batch, time, num_input)
            y_train_soma_scale: Soma voltage scaling factor

        Returns:
            tuple of (outputs, synapse_record, memory_record):
            - outputs: (batch, time, 2) postprocessed predictions
            - synapse_record: (batch, time, num_synapse) traces
            - memory_record: (batch, time, num_memory) states
        """
        batch_size, T, _ = X.shape

        # Cache values
        w_s = self.w_s
        kappa_s, kappa_m = self.kappa_s, self.kappa_m

        # Initialize states
        s_prev = torch.zeros(batch_size, len(kappa_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Accumulate outputs and states
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        s_record = torch.jit.annotate(list[torch.Tensor], [])
        m_record = torch.jit.annotate(list[torch.Tensor], [])

        inputs = self.route_input_to_synapses(X)

        # Process each timestep, recording internal states
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)
            s_record.append(s_prev)
            m_record.append(m_prev)

        # Stack along time dimension
        outputs = torch.stack(outputs, dim=-2)
        s_record = torch.stack(s_record, dim=-2)
        m_record = torch.stack(m_record, dim=-2)

        # Postprocess outputs
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = (1 / y_train_soma_scale) * soma_pred
        outputs = torch.stack([spike_pred, soma_pred], dim=-1)

        return outputs, s_record, m_record
