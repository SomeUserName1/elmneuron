"""Expressive Leaky Memory (ELM) Neuron - Version 2.

This module implements the ELM v2 neuron architecture, an improved version of
the biologically-inspired recurrent neural network unit for processing temporal
spike-based data.

Key improvements in v2 over v1:
    - Removed explicit synapse states for improved efficiency
    - Branch states now handle leaky integration directly from weighted inputs
    - Modified memory update formula: m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t
      where κ_λ = exp(-Δt * λ / τ_m), providing better numerical stability
    - Reduced computational cost and memory footprint
    - Maintained biological interpretability with simplified dynamics

The ELM v2 neuron features:
    - Direct weighted input integration into branch states
    - Intermediate branch layer with leaky dynamics (replaces synapse layer)
    - Slow memory dynamics with heterogeneous time constants
    - Multi-timescale processing for capturing both transient and persistent
      patterns

Architectural differences from v1:
    - Two-stage processing (branch → memory) instead of three-stage
      (synapse → branch → memory)
    - Branch time constants (τ_b) replace synapse time constants (τ_s)
    - More stable memory update with kappa_lambda formulation
    - Branches directly compute leaky integration:
      b_t = κ_b * b_{t-1} + w_s * x_t

Typical usage:
    >>> model = ELM(num_input=128, num_output=2, num_memory=100)
    >>> outputs = model(inputs)  # inputs: (batch, time, features)

For more details on the architecture and mathematical formulation, see the
class docstring for `ELM`.
"""

import math

import torch
import torch.nn as nn
from torch import jit
from torch.nn.parameter import Parameter

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
    """Expressive Leaky Memory (ELM) neuron model (v2).

    A biologically-inspired recurrent neural network unit with improved
    efficiency and numerical stability. Version 2 simplifies the dynamics
    while maintaining multi-timescale temporal processing capabilities.

    Architecture:
        1. Input Layer: Routes and weights input signals to branches
        2. Branch Layer: Leaky integration of weighted inputs with fast
                         dynamics
        3. Memory Layer: Slow-decaying states that maintain long-term
                         dependencies
        4. Output Layer: Linear projection from memory states to outputs

    Dynamics:
        At each timestep t, the model computes:

        Branch update (new in v2 - direct from inputs):
            b_inp = Σ_{j∈branch_i} (w_s[j] * x_t[j])
            b_t = κ_b * b_{t-1} + b_inp

        Memory update (improved stability):
            Δm_t = tanh(MLP([b_t; κ_m * m_{t-1}]))
            m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t

        Output:
            y_t = W_y * m_t

    where:
        - κ_b = exp(-Δt / τ_b) is the branch decay factor
        - κ_m = exp(-Δt / τ_m) is the memory decay factor
        - κ_λ = exp(-Δt * λ / τ_m) is the memory update decay factor
        - w_s are learnable synaptic weights (non-negative)
        - MLP is a multi-layer perceptron
        - τ_b, τ_m are time constants controlling decay rates

    Key differences from v1:
        - No separate synapse states (s_t removed)
        - Branch states directly integrate weighted inputs
        - Memory update uses (1 - κ_λ) instead of λ * (1 - κ_m)
        - Better numerical stability for large λ values
        - Reduced memory and computation requirements

    Args:
        num_input: Number of input features/channels
        num_output: Number of output features/channels
        num_memory: Number of memory units (slow dynamics). Default: 100
        lambda_value: Memory update scaling factor (λ). Controls memory
                      plasticity. Larger values → faster memory updates.
                      Default: 5.0
        mlp_num_layers: Number of hidden layers in the MLP. Default: 1
        mlp_hidden_size: Hidden dimension of MLP layers. If None,
                         defaults to 2*num_memory
        mlp_activation: Activation function ("relu" or "silu"). Default: "relu"
        tau_b_value: Branch time constant in timesteps. Controls branch decay
                     speed. Larger values → slower decay → longer branch mem.
                     Default: 5.0
        memory_tau_min: Minimum memory time constant in timesteps. Default: 1.0
        memory_tau_max: Maximum memory time constant in timesteps.
                        Default: 1000.0
        learn_memory_tau: Whether memory time constants are learnable.
                          Default: False
        w_s_value: Initial value for synaptic weights. Default: 0.5
        num_branch: Number of branches (groups of synapses).
                    If None, defaults to num_input
        num_synapse_per_branch: Number of synapses per branch. Default: 1
        input_to_synapse_routing: Input routing strategy. Options:
            - None: Direct 1-to-1 mapping (requires num_synapse == num_input)
            - "random_routing": Random sparse connectivity
            - "neuronio_routing": Biological routing with interlaced E/I and
                                  overlapping windows
            Default: None
        delta_t: Temporal resolution (timestep size). Default: 1.0

    Example:
        >>> # Basic usage
        >>> model = ELM(num_input=128, num_output=2, num_memory=100)
        >>> inputs = torch.randn(32, 1000, 128)  # (batch, time, features)
        >>> outputs = model(inputs)  # (32, 1000, 2)
        >>>
        >>> # Custom configuration with routing
        >>> model = ELM(
        ...     num_input=256,
        ...     num_output=1,
        ...     num_memory=200,
        ...     lambda_value=10.0,
        ...     num_branch=64,
        ...     num_synapse_per_branch=4,
        ...     input_to_synapse_routing="random_routing"
        ... )
    """

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
    ):
        super(ELM, self).__init__()

        # Store basic configuration parameters
        self.num_input: int = num_input
        self.num_output: int = num_output
        self.num_memory: int = num_memory
        self.lambda_value: float = lambda_value
        self.mlp_num_layers: int = mlp_num_layers
        self.mlp_activation: str = mlp_activation
        self.memory_tau_min: float = memory_tau_min
        # Ensure memory_tau_max is strictly greater than memory_tau_min
        # Small epsilon prevents numerical issues in scaled_sigmoid
        self.memory_tau_max: float = max(memory_tau_min + 3e-6, memory_tau_max)
        self.learn_memory_tau: bool = learn_memory_tau
        self.tau_b_value: float = tau_b_value
        self.w_s_value: float = w_s_value
        self.num_synapse_per_branch: int = num_synapse_per_branch
        self.input_to_synapse_routing: str | None = input_to_synapse_routing
        self.delta_t: float = delta_t

        # Derive neuron architecture properties
        # MLP hidden size defaults to 2x memory dimension if not specified
        self.mlp_hidden_size: int = (
            mlp_hidden_size if mlp_hidden_size else 2 * num_memory
        )
        # Number of branches defaults to number of inputs (1-to-1 mapping)
        self.num_branch: int = self.num_input if num_branch is None else num_branch
        # MLP receives concatenated branch activities and memory states
        self.num_mlp_input: int = self.num_branch + num_memory
        # Total synapses = synapses per branch × number of branches
        self.num_synapse: int = num_synapse_per_branch * self.num_branch

        # Validate configuration
        # Either num_synapse matches num_input (direct mapping) or routing
        # must be specified
        assert self.num_synapse == num_input or input_to_synapse_routing is not None
        assert self.input_to_synapse_routing in PREPROCESS_CONFIGURATIONS

        # Initialize learnable components
        # MLP: maps [branch_states; memory_states] → memory_updates
        self.mlp: MLP = MLP(
            self.num_mlp_input,
            self.mlp_hidden_size,
            num_memory,
            mlp_num_layers,
            mlp_activation,
        )
        # Synaptic weights: will be constrained to non-negative via ReLU
        # in w_s property
        # Stored as "proto" (prototype) to enable unconstrained optimization
        self._proto_w_s: Parameter = Parameter(
            torch.full((self.num_synapse,), w_s_value)
        )
        # Output projection: maps memory states to output space
        self.w_y: nn.Linear = nn.Linear(num_memory, num_output)

        # Initialize branch time constants (fixed, not learned)
        # All branches share the same time constant in this implementation
        tau_b: torch.Tensor = torch.full((self.num_branch,), tau_b_value)
        self.tau_b: Parameter = Parameter(tau_b, requires_grad=False)

        # Initialize memory time constants (optionally learnable)
        # Log-spaced distribution across [memory_tau_min, memory_tau_max]
        # Creates heterogeneous timescales for multi-scale temporal processing
        # Fast units (small τ) capture transient patterns
        # Slow units (large τ) maintain long-term dependencies
        _proto_tau_m: torch.Tensor = torch.logspace(
            math.log10(self.memory_tau_min + 1e-6),
            math.log10(self.memory_tau_max - 1e-6),
            num_memory,
        )
        # Transform to unconstrained space for optimization
        # inverse_scaled_sigmoid enables gradient-based learning while ensuring
        # values stay in [memory_tau_min, memory_tau_max] after forward pass
        _proto_tau_m = inverse_scaled_sigmoid(
            _proto_tau_m, self.memory_tau_min, self.memory_tau_max
        )
        self._proto_tau_m: Parameter = Parameter(
            _proto_tau_m, requires_grad=learn_memory_tau
        )

        # Create input-to-synapse routing indices
        # Stored as parameters for proper device transfer and model
        # serialization
        routing_artifacts: tuple[torch.Tensor | None, torch.Tensor | None] = (
            self.create_input_to_synapse_indices()
        )
        # indices[i] specifies which input channel feeds synapse i
        self.input_to_synapse_indices: Parameter = Parameter(
            routing_artifacts[0], requires_grad=False
        )
        # Binary mask for valid connections (handles padding in overlapping
        # routing)
        self.valid_indices_mask: Parameter = Parameter(
            routing_artifacts[1], requires_grad=False
        )

    @property
    def tau_m(self) -> torch.Tensor:
        """Memory time constants in timesteps.

        Returns:
            torch.Tensor: Shape (num_memory,),
                          values in [memory_tau_min, memory_tau_max].
                Longer time constants → slower decay → longer-lasting memory.
                Log-spaced distribution enables multi-timescale processing.
        """
        return scaled_sigmoid(
            self._proto_tau_m, self.memory_tau_min, self.memory_tau_max
        )

    @property
    def kappa_m(self) -> torch.Tensor:
        """Memory decay factors.

        Returns:
            torch.Tensor: Shape (num_memory,), values in (0, 1).
                Computed as κ_m = exp(-Δt / τ_m).
                Higher values → slower decay → more persistent memory.
                Used for exponential leaky integration:
                    m_t = κ_m * m_{t-1} + ...
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))

    @property
    def kappa_lambda(self) -> torch.Tensor:
        """Memory update decay factors (v2 improvement).

        This is a key change in v2 that improves numerical stability.
        Instead of using λ * (1 - κ_m) for memory updates, v2 uses (1 - κ_λ)
        where κ_λ incorporates both the time constant and lambda scaling.

        Returns:
            torch.Tensor: Shape (num_memory,), values in (0, 1).
                Computed as κ_λ = exp(-Δt * λ / τ_m).
                Controls the magnitude of memory updates in:
                    m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t
                Larger λ → smaller κ_λ → larger updates (more plasticity)
                Prevents overflow issues when λ is large
        """
        clamped_tau: float = torch.clamp(self.tau_m, min=1e-6)
        return torch.exp(-self.delta_t * self.lambda_value / clamped_tau)

    @property
    def kappa_b(self):
        """Branch decay factors.

        Returns:
            torch.Tensor: Shape (num_branch,), values in (0, 1).
                Computed as κ_b = exp(-Δt / τ_b).
                Higher values → slower decay → longer branch integration
                window.
                Controls branch dynamics: b_t = κ_b * b_{t-1} + b_inp
        """
        return torch.exp(-self.delta_t / torch.clamp(self.tau_b, min=1e-6))

    @property
    def w_s(self) -> torch.Tensor:
        """Synaptic weights (non-negative).

        Returns:
            torch.Tensor: Shape (num_synapse,), values >= 0.
                Learnable weights constrained to non-negative via ReLU
                activation.
                Applied element-wise to inputs before branch aggregation.
        """
        return torch.relu(self._proto_w_s)

    def create_input_to_synapse_indices(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Create indices for routing inputs to synapses.

        Generates routing patterns that determine how input channels are
        connected to synapses. Supports multiple routing strategies for
        different use cases.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                - input_to_synapse_indices: Shape (num_synapse,). Specifies
                    which input channel feeds each synapse. None if no routing
                    is used.
                - valid_indices_mask: Shape (num_synapse,). Binary mask where
                    1 indicates a valid connection. Handles padding in
                    overlapping windows. None if no routing is used.

        Routing strategies:
            - None: Direct 1-to-1 mapping (no routing needed)
            - "random_routing": Random sparse connectivity
                - Each synapse randomly samples one input channel
                - Creates diverse, unstructured receptive fields
                - All connections are valid (no masking)
            - "neuronio_routing": Biologically-inspired routing for NeuronIO
                                  dataset
                - Step 1: Interlace excitatory (even) and inhibitory (odd)
                    inputs. Pattern: [0, N/2, 1, N/2+1, 2, N/2+2, ...]
                - Step 2: Create overlapping sliding windows
                    Each branch receives inputs from a local window
                - Combines balanced E/I input with spatial locality
                - May include padding (handled by valid_indices_mask)
        """
        if self.input_to_synapse_routing == "random_routing":
            # Randomly select num_synapse input channels from num_input total
            # channels. Each synapse gets one randomly chosen input
            input_to_synapse_indices: torch.Tensor = torch.randint(
                self.num_input, (self.num_synapse,)
            )
            # All connections are valid (no masking needed)
            return (input_to_synapse_indices, torch.ones_like(input_to_synapse_indices))

        elif self.input_to_synapse_routing == "neuronio_routing":
            # Validate that windows can accommodate all inputs
            # Ensures no inputs are dropped due to insufficient window size
            assert (
                math.ceil(self.num_input / self.num_branch)
                <= self.num_synapse_per_branch
            ), "num_synapse_per_branch too small for neuronio_routing"

            # Step 1: Interlace excitatory and inhibitory inputs
            # Balances E/I inputs across branches
            # Pattern: [0, N/2, 1, N/2+1, 2, N/2+2, ...]
            interlocking_indices = create_interlocking_indices(self.num_input)

            # Step 2: Create overlapping sliding windows
            # Assigns neighboring inputs to the same branch
            # Provides spatial locality and receptive field overlap
            overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
                self.num_input, self.num_branch, self.num_synapse_per_branch
            )

            # Combine: apply interlocking to windowed indices
            # Results in branches with balanced E/I and local receptive fields
            input_to_synapse_indices = interlocking_indices[overlapping_indices]

            return input_to_synapse_indices, valid_indices_mask
        else:
            # No routing: assume direct 1-to-1 mapping
            return None, None

    def route_input_to_synapses(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input-to-synapse routing transformation.

        Routes input channels to synapses according to the configured routing
        strategy.
        This implements the connectivity pattern between inputs and synapses.

        Args:
            x: Input tensor, shape (batch_size, T, num_input)

        Returns:
            torch.Tensor: Routed inputs, shape (batch_size, T, num_synapse)
                If no routing: returns x unchanged
                    (requires num_synapse == num_input)
                If routing: selects and reorders input channels according
                    to indices
        """
        if self.input_to_synapse_routing is not None:
            # Select input channels according to routing indices
            # input_to_synapse_indices[i] specifies which input feeds synapse i
            x = torch.index_select(x, 2, self.input_to_synapse_indices)
            # Apply validity mask (zeros out padded connections in overlapping
            # windows)
            x = x * self.valid_indices_mask
        return x

    @jit.script_method
    def dynamics(
        self,
        x: torch.Tensor,
        b_prev: torch.Tensor,
        m_prev: torch.Tensor,
        w_s: torch.Tensor,
        kappa_b: torch.Tesnor,
        kappa_m: torch.Tensor,
        kappa_lambda: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ELM v2 dynamics for a single timestep.

        This is the core recurrent update implementing the two-stage
        computation: input → branch → memory → output.
        Key improvement over v1: branches directly integrate weighted inputs
        without separate synapse states.

        Args:
            x: Input at time t, shape (batch_size, num_synapse)
                Already routed from original inputs via route_input_to_synapses
            b_prev: Previous branch states, shape (batch_size, num_branch)
            m_prev: Previous memory states, shape (batch_size, num_memory)
            w_s: Synaptic weights, shape (num_synapse,)
            kappa_b: Branch decay factors, shape (num_branch,)
            kappa_m: Memory decay factors, shape (num_memory,)
            kappa_lambda: Memory update decay factors, shape (num_memory,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - y_t: Output at time t, shape (batch_size, num_output)
                - b_t: Updated branch states, shape (batch_size, num_branch)
                - m_t: Updated memory states, shape (batch_size, num_memory)
        """
        batch_size: int = x.shape[0]

        # Step 1: Branch input computation
        # Apply synaptic weights element-wise to inputs, then aggregate per
        # branch
        # w_s * x: shape (batch_size, num_synapse) - weighted inputs
        # .view(...): reshape to (batch_size, num_branch,
        #                         num_synapse_per_branch)
        # .sum(dim=-1): sum synapses within each branch → (batch_size,
        #                                                  num_branch)
        b_inp = (w_s * x).view(batch_size, self.num_branch, -1).sum(dim=-1)

        # Step 2: Branch dynamics (leaky integration)
        # b_t = κ_b * b_{t-1} + b_inp
        # Exponential decay of previous state + new input
        b_t = kappa_b * b_prev + b_inp

        # Step 3: Memory update computation
        # MLP receives: [current branches; decayed memories]
        # Δm_t = tanh(MLP([b_t; κ_m * m_{t-1}]))
        # custom_tanh provides numerically stable tanh activation
        delta_m_t = custom_tanh(self.mlp(torch.cat([b_t, kappa_m * m_prev], dim=-1)))

        # Step 4: Memory dynamics (v2 formulation)
        # m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t
        # Key v2 improvement: uses (1 - κ_λ) instead of λ * (1 - κ_m)
        # where κ_λ = exp(-Δt * λ / τ_m)
        # This prevents numerical overflow when λ is large
        m_t = kappa_m * m_prev + (1 - kappa_lambda) * delta_m_t

        # Step 5: Output projection
        # Linear mapping from memory states to output space
        y_t = self.w_y(m_t)

        return y_t, b_t, m_t

    @jit.script_method
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ELM v2 neuron.

        Processes a batch of input sequences by iteratively applying the
        dynamics function for each timestep. Branch and memory states are
        initialized to zero at the start of each sequence.

        Args:
            X: Input sequences, shape (batch_size, T, num_input)
                - batch_size: Number of sequences in the batch
                - T: Sequence length (number of timesteps)
                - num_input: Input feature dimension

        Returns:
            torch.Tensor: Output sequences, shape (batch_size, T, num_output)
                Raw model outputs (no post-processing applied)
        """
        batch_size, T = X.shape[:2]

        # Get current parameter values
        w_s = self.w_s
        kappa_b = self.kappa_b
        kappa_m = self.kappa_m
        kappa_lambda = self.kappa_lambda

        # Initialize branch and memory states to zero
        b_prev = torch.zeros(batch_size, len(kappa_b), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Collect outputs for each timestep
        outputs = torch.jit.annotate(list[torch.Tensor], [])

        # Apply input routing (if configured)
        inputs = self.route_input_to_synapses(X)

        # Iterate through sequence, updating states at each timestep
        for t in range(T):
            y_t, b_prev, m_prev = self.dynamics(
                inputs[:, t], b_prev, m_prev, w_s, kappa_b, kappa_m, kappa_lambda
            )
            outputs.append(y_t)

        # Stack outputs: list of (batch, num_output) → (batch, T, num_output)
        return torch.stack(outputs, dim=-2)

    @jit.script_method
    def neuronio_eval_forward(
        self, X: torch.Tesnor, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ) -> torch.Tensor:
        """Forward pass for NeuronIO evaluation with post-processing.

        Specialized forward pass for the NeuronIO dataset that applies
        appropriate activations and scaling to raw model outputs. NeuronIO is
        a biological neuroscience dataset where the task is to predict
        neuronal spikes and soma voltage from synaptic inputs.

        Args:
            X: Input sequences, shape (batch_size, T, num_input)
                Typically represents synaptic input currents to a neuron
            y_train_soma_scale: Scaling factor used during training for soma
                                voltage.
                During training, soma voltages are scaled down for numerical
                stability. This parameter reverses that scaling for evaluation.
                Default: DEFAULT_Y_TRAIN_SOMA_SCALE

        Returns:
            torch.Tensor: Predictions, shape (batch_size, T, 2) where:
                - [..., 0]: Spike probabilities in [0, 1] (sigmoid-activated)
                    Binary prediction of whether the neuron fires an action
                    potential
                - [..., 1]: Soma voltage predictions (rescaled to original mV
                  units). Continuous prediction of membrane potential
        """
        # Get raw outputs from model
        outputs = self.forward(X)
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]

        # Apply sigmoid to spike prediction → probability in [0, 1]
        spike_pred = torch.sigmoid(spike_pred)

        # Rescale soma prediction back to original voltage units (typically mV)
        soma_pred = 1 / y_train_soma_scale * soma_pred

        return torch.stack([spike_pred, soma_pred], dim=-1)

    @jit.script_method
    def neuronio_viz_forward(
        self, X: torch.Tensor, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with internal state recording for visualization.

        Similar to neuronio_eval_forward but also returns the complete history
        of internal branch and memory states for analysis and visualization.
        Useful for understanding model dynamics and debugging.

        Note: In v2, branch states replace synapse states from v1.

        Args:
            X: Input sequences, shape (batch_size, T, num_input)
            y_train_soma_scale: Scaling factor used during training for soma
                voltage. Default: DEFAULT_Y_TRAIN_SOMA_SCALE

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - outputs: Predictions, shape (batch_size, T, 2)
                    - [..., 0]: Spike probabilities (post-sigmoid)
                    - [..., 1]: Soma voltage predictions (rescaled)
                - b_record: Branch state history, shape (batch_size, T,
                                                         num_branch)
                    Full trajectory of branch activations over time
                - m_record: Memory state history, shape (batch_size, T,
                                                         num_memory)
                    Full trajectory of memory activations over time

        Note:
            This method is more memory-intensive than neuronio_eval_forward
            due to storing all internal states. Use only when visualization or
            analysis of internal dynamics is needed.
        """
        batch_size, T = X.shape[:2]

        # Get current parameter values
        w_s = self.w_s
        kappa_b = self.kappa_b
        kappa_m = self.kappa_m
        kappa_lambda = self.kappa_lambda

        # Initialize branch and memory states to zero
        b_prev = torch.zeros(batch_size, len(kappa_b), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # Prepare storage for outputs and internal states
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        b_record = torch.jit.annotate(list[torch.Tensor], [])
        m_record = torch.jit.annotate(list[torch.Tensor], [])

        # Apply input routing
        inputs = self.route_input_to_synapses(X)

        # Iterate through sequence, recording states at each timestep
        for t in range(T):
            y_t, b_prev, m_prev = self.dynamics(
                inputs[:, t], b_prev, m_prev, w_s, kappa_b, kappa_m, kappa_lambda
            )
            outputs.append(y_t)
            b_record.append(b_prev)  # Record branch states
            m_record.append(m_prev)  # Record memory states

        # Stack all recorded values
        outputs = torch.stack(outputs, dim=-2)
        b_record = torch.stack(b_record, dim=-2)
        m_record = torch.stack(m_record, dim=-2)

        # Post-process outputs (same as neuronio_eval_forward)
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = 1 / y_train_soma_scale * soma_pred
        outputs = torch.stack([spike_pred, soma_pred], dim=-1)

        return outputs, b_record, m_record
