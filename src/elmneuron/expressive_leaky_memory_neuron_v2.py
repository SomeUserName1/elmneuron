"""
Expressive Leaky Memory (ELM) Neuron Model - Version 2 (Branch-ELM).

This module implements the Branch-ELM neuron, an improved variant
that achieves ~7× parameter reduction over v1 through hierarchical
processing mimicking dendritic tree structure. The model first
reduces synaptic inputs to branch activations, then processes these
through an MLP for memory updates.

Key architectural components:
1. Branch dynamics: Weighted aggregation of synaptic inputs per branch
2. Integration mechanism: MLP processes branch inputs + memory
3. Memory dynamics: Modified leaky integrators for improved stability
4. Output dynamics: Linear readout from memory state

**Main improvements over v1**:
- 7× fewer parameters (8K vs 53K for NeuronIO)
- More stable training (modified memory update equation)
- Learnable synapse weights (crucial for expressivity)
- Biologically plausible dendritic tree structure

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
    Branch-ELM (Expressive Leaky Memory) neuron model (Version 2).

    The Branch-ELM achieves significant parameter reduction through
    two-stage hierarchical processing: synaptic inputs are first
    aggregated into branch activations (mimicking dendritic branches),
    then processed via an MLP for memory updates.

    **Key differences from ELM v1**:
    - Two-stage processing: synapse → branch → MLP
    - Learnable synapse weights (w_s must be trained)
    - Branch state (b_t) instead of synapse state (s_t)
    - Modified memory update: m_t = κ_m*m_{t-1} + (1-κ_λ)*Δm_t
    - Additional decay factor: κ_λ = exp(-Δt*λ/τ_m)
    - Improved stability, especially with λ = 5.0

    **Architecture flow**:
        x_t → [weighted sum per branch] → b_t →
        [MLP(b_t, m_{t-1})] → Δm_t →
        [modified memory update] → m_t →
        [linear readout] → y_t

    **Routing requirement**:
        Branch-ELM typically requires 'neuronio_routing' to properly
        assign inputs to branches. Must oversample (more synapses
        than inputs) for best performance.

    Args:
        num_input: Input dimension (number of synaptic inputs)
        num_output: Output dimension
        num_memory: Number of memory units (default: 100)
            Typical values: 15-20 for NeuronIO, 100 for long tasks
        lambda_value: Memory update scaling factor (default: 5.0)
            Used in κ_λ calculation for improved stability
        mlp_num_layers: Number of MLP hidden layers (default: 1)
        mlp_hidden_size: MLP hidden dimension (default: 2*num_memory)
        mlp_activation: MLP activation function ('relu' or 'silu')
        tau_b_value: Branch time constant in ms (default: 5.0)
            Replaces tau_s from v1
        memory_tau_min: Min memory timescale in ms (default: 1.0)
        memory_tau_max: Max memory timescale in ms (default: 1000.0)
            Adjusted to max(tau_min + 3e-6, tau_max) for stability
        learn_memory_tau: Whether to learn memory timescales
        w_s_value: Initial synapse weight value (default: 0.5)
            **Important**: Unlike v1, these MUST be learnable
        num_branch: Number of branches (default: num_input)
        num_synapse_per_branch: Synapses per branch (default: 1)
        input_to_synapse_routing: Routing strategy
            (None, 'random_routing', 'neuronio_routing')
            Typically requires 'neuronio_routing'
        delta_t: Fictitious timestep in ms (default: 1.0)

    Attributes:
        tau_m: Memory timescales (property, bounded)
        kappa_m: Memory decay factors = exp(-delta_t / tau_m)
        kappa_b: Branch decay factors = exp(-delta_t / tau_b)
        kappa_lambda: Combined decay = exp(-delta_t * lambda / tau_m)
        w_s: Synapse weights (property, ReLU-activated, LEARNABLE)

    Example:
        >>> model = ELM(
        ...     num_input=1278,
        ...     num_output=2,
        ...     num_memory=15,
        ...     lambda_value=5.0,
        ...     num_branch=45,
        ...     num_synapse_per_branch=100,
        ...     input_to_synapse_routing='neuronio_routing'
        ... )
        >>> X = torch.randn(8, 100, 1278)  # (batch, time, input)
        >>> Y = model(X)  # (batch, time, output)
    """

    # JIT script constants for compilation
    __constants__: list[str] = [
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
        tau_b_value: float = 5.0,
        memory_tau_min: float = 1.0,
        memory_tau_max: float = 1000.0,
        learn_memory_tau: bool = False,
        w_s_value: float = 0.5,
        num_branch: int | None = None,
        num_synapse_per_branch: int = 1,
        input_to_synapse_routing: str | None = None,
        delta_t: float = 1.0,
    ) -> None:
        """Initialize the Branch-ELM neuron model."""
        super(ELM, self).__init__()

        # Store basic dimensions
        self.num_input = num_input
        self.num_output = num_output
        self.num_memory = num_memory
        self.lambda_value = lambda_value
        self.mlp_num_layers = mlp_num_layers
        self.mlp_activation = mlp_activation

        # Store timescale bounds
        # Note: tau_max adjusted for numerical stability
        self.memory_tau_min = memory_tau_min
        self.memory_tau_max = max(memory_tau_min + 3e-6, memory_tau_max)
        self.learn_memory_tau = learn_memory_tau

        # Store branch/synapse parameters
        # Note: tau_b replaces tau_s from v1
        self.tau_b_value = tau_b_value
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

        # Initialize synapse weights (LEARNABLE in v2, critical!)
        # ReLU activation ensures positivity
        # These weights perform input gating at branch level
        self._proto_w_s = nn.parameter.Parameter(
            torch.full((self.num_synapse,), w_s_value)
        )

        # Initialize output layer (linear readout from memory)
        self.w_y = nn.Linear(num_memory, num_output)

        # Initialize branch time constants (fixed)
        # Note: branches (not synapses) have timescales in v2
        tau_b = torch.full((self.num_branch,), tau_b_value)
        self.tau_b = nn.parameter.Parameter(tau_b, requires_grad=False)

        # Initialize memory time constants
        # Logspace distribution ensures diverse timescales
        _proto_tau_m = torch.logspace(
            math.log10(self.memory_tau_min + 1e-6),
            math.log10(self.memory_tau_max - 1e-6),
            num_memory,
        )
        # Apply inverse scaled sigmoid for bounded optimization
        _proto_tau_m = inverse_scaled_sigmoid(
            _proto_tau_m, self.memory_tau_min, self.memory_tau_max
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
    def kappa_lambda(self) -> torch.Tensor:
        """
        Combined decay factor for modified memory update (v2).

        Computed as exp(-delta_t * lambda / tau_m), this combines
        the effects of λ and memory timescales for improved
        stability. This is a key difference from v1.

        Used in memory update: m_t = κ_m*m_{t-1} + (1-κ_λ)*Δm_t

        Returns:
            Tensor of shape (num_memory,) with values in (0, 1)
        """
        return torch.exp(
            -self.delta_t * self.lambda_value / torch.clamp(self.tau_m, min=1e-6)
        )

    @property
    def kappa_b(self) -> torch.Tensor:
        """
        Branch decay factors.

        Computed as exp(-delta_t / tau_b), representing the
        fraction of previous branch activation retained.

        Returns:
            Tensor of shape (num_branch,) with values in (0, 1)
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_b, min=1e-6))

    @property
    def w_s(self) -> torch.Tensor:
        """
        Synapse weights (non-negative via ReLU, LEARNABLE).

        **Critical difference from v1**: These weights MUST be
        learnable in Branch-ELM. They cannot be absorbed into the
        MLP due to the branch reduction step.

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

        For Branch-ELM, 'neuronio_routing' is typically required
        to exploit spatial structure in input organization.

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
            # (exploits spatial locality in dendritic tree)
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
        b_prev: torch.Tensor,
        m_prev: torch.Tensor,
        w_s: torch.Tensor,
        kappa_b: torch.Tensor,
        kappa_m: torch.Tensor,
        kappa_lambda: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Branch-ELM dynamics for a single timestep.

        This is the core computation implementing:
        1. Branch input: weighted sum of synapses per branch
        2. Branch dynamics: b_t = κ_b * b_{t-1} + branch_input
        3. Memory proposal: Δm_t = tanh(MLP([b_t, κ_m*m_{t-1}]))
        4. Modified memory update: m_t = κ_m*m_{t-1} + (1-κ_λ)*Δm_t
        5. Output: y_t = W_y * m_t

        **Key difference from v1**: Branch-level processing and
        modified memory update using κ_λ instead of λ*(1-κ_m).

        Args:
            x: Current input (batch, num_synapse)
            b_prev: Previous branch state (batch, num_branch)
            m_prev: Previous memory state (batch, num_memory)
            w_s: Synapse weights (num_synapse,)
            kappa_b: Branch decay factors (num_branch,)
            kappa_m: Memory decay factors (num_memory,)
            kappa_lambda: Combined decay factors (num_memory,)

        Returns:
            tuple of (y_t, b_t, m_t):
            - y_t: Output (batch, num_output)
            - b_t: Updated branch state (batch, num_branch)
            - m_t: Updated memory state (batch, num_memory)
        """
        batch_size, _ = x.shape

        # Compute weighted branch input
        # Each branch receives weighted sum of its synapses
        # (w_s * x_t per synapse, then sum per branch)
        b_inp = (w_s * x).view(batch_size, self.num_branch, -1).sum(dim=-1)

        # Update branch activations (exponential filtering)
        # b_t = κ_b * b_{t-1} + branch_input
        b_t = kappa_b * b_prev + b_inp

        decayed_m_prev = kappa_m * m_prev
        # Compute memory update proposal via MLP
        # Input: [branch_activations, decayed_previous_memory]
        # Output: Δm_t (memory update proposal)
        delta_m_t = custom_tanh(self.mlp(torch.cat([b_t, decayed_m_prev], dim=-1)))

        # Update memory with modified equation (v2 improvement)
        # m_t = forget * m_{t-1} + input_scale * update
        # where input_scale = (1 - κ_λ) ensures stability
        # This replaces v1's λ * (1 - κ_m) formulation
        m_t = decayed_m_prev + (1 - kappa_lambda) * delta_m_t

        # Compute output via linear readout
        y_t = self.w_y(m_t)

        return y_t, b_t, m_t

    @jit.script_method
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence through the Branch-ELM neuron.

        Iteratively applies dynamics() for each timestep,
        maintaining branch and memory states across time.

        Args:
            X: Input sequence (batch, time, num_input)

        Returns:
            Output sequence (batch, time, num_output)
        """
        batch_size, T, _ = X.shape

        # Cache frequently used values
        w_s = self.w_s
        kappa_b, kappa_m = self.kappa_b, self.kappa_m
        kappa_lambda = self.kappa_lambda

        # Initialize states to zero
        b_prev = torch.zeros(batch_size, len(kappa_b), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Accumulate outputs (JIT-compatible list annotation)
        outputs = torch.jit.annotate(list[torch.Tensor], [])

        # Apply input routing once for efficiency
        inputs = self.route_input_to_synapses(X)

        # Process each timestep
        for t in range(T):
            y_t, b_prev, m_prev = self.dynamics(
                inputs[:, t], b_prev, m_prev, w_s, kappa_b, kappa_m, kappa_lambda
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
            tuple of (outputs, branch_record, memory_record):
            - outputs: (batch, time, 2) postprocessed predictions
            - branch_record: (batch, time, num_branch) activations
            - memory_record: (batch, time, num_memory) states
        """
        batch_size, T, _ = X.shape

        # Cache values
        w_s = self.w_s
        kappa_b, kappa_m = self.kappa_b, self.kappa_m
        kappa_lambda = self.kappa_lambda

        # Initialize states
        b_prev = torch.zeros(batch_size, len(kappa_b), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Accumulate outputs and states
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        b_record = torch.jit.annotate(list[torch.Tensor], [])
        m_record = torch.jit.annotate(list[torch.Tensor], [])

        inputs = self.route_input_to_synapses(X)

        # Process each timestep, recording internal states
        for t in range(T):
            y_t, b_prev, m_prev = self.dynamics(
                inputs[:, t], b_prev, m_prev, w_s, kappa_b, kappa_m, kappa_lambda
            )
            outputs.append(y_t)
            b_record.append(b_prev)
            m_record.append(m_prev)

        # Stack along time dimension
        outputs = torch.stack(outputs, dim=-2)
        b_record = torch.stack(b_record, dim=-2)
        m_record = torch.stack(m_record, dim=-2)

        # Postprocess outputs
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = (1 / y_train_soma_scale) * soma_pred
        outputs = torch.stack([spike_pred, soma_pred], dim=-1)

        return outputs, b_record, m_record
