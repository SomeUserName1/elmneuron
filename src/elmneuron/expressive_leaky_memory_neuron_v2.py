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

**Lightning Version**:
This version uses PyTorch Lightning for training and torch.compile
for performance instead of TorchScript JIT. The model is dataset-agnostic
and routing logic has been moved to data transforms for better modularity.

Reference:
    Spieler, A., Rahaman, N., Martius, G., Schölkopf, B., &
    Levina, A. (2023). The ELM Neuron: an Efficient and Expressive
    Cortical Neuron Model Can Solve Long-Horizon Tasks.
    arXiv preprint arXiv:2306.16922.
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .modeling_utils import MLP, custom_tanh, inverse_scaled_sigmoid, scaled_sigmoid


class ELM(pl.LightningModule):
    """
    Branch-ELM (Expressive Leaky Memory) neuron model (Version 2) - Lightning Edition.

    A dataset-agnostic PyTorch Lightning module that implements hierarchical
    dendritic processing. The model processes sequential data through:
    1. Weighted aggregation of inputs into branch activations
    2. MLP-based integration of branch and memory states
    3. Modified leaky memory updates for stability
    4. Linear readout for predictions

    **Key differences from ELM v1**:
    - Two-stage processing: input → branch → memory
    - Learnable synapse weights (w_s must be trained)
    - Branch state (b_t) instead of synapse state (s_t)
    - Modified memory update: m_t = κ_m*m_{t-1} + (1-κ_λ)*Δm_t
    - Improved stability, especially with λ = 5.0

    **Architecture flow**:
        x_t → [weighted sum per branch] → b_t →
        [MLP(b_t, decayed_m_{t-1})] → Δm_t →
        [modified memory update] → m_t →
        [linear readout] → y_t

    **Dataset-agnostic design**:
        This model accepts generic sequential input of shape (batch, time, num_input)
        and produces output of shape (batch, time, num_output). Input routing,
        sequentialization of non-sequential data, and task-specific logic are
        handled by data transforms and task-specific wrapper modules.

    Args:
        num_input: Input dimension
        num_output: Output dimension
        num_branch: Number of branches (default: num_input)
        num_synapse_per_branch: Synapses per branch (default: 1)
        num_memory: Number of memory units (default: 100)
        lambda_value: Memory update scaling factor (default: 5.0)
        mlp_num_layers: Number of MLP hidden layers (default: 1)
        mlp_hidden_size: MLP hidden dimension (default: 2*num_memory)
        mlp_activation: MLP activation ('relu' or 'silu', default: 'relu')
        tau_b_value: Branch time constant in ms (default: 5.0)
        memory_tau_min: Min memory timescale in ms (default: 1.0)
        memory_tau_max: Max memory timescale in ms (default: 1000.0)
        learn_memory_tau: Whether to learn memory timescales (default: False)
        w_s_value: Initial synapse weight value (default: 0.5)
        delta_t: Fictitious timestep in ms (default: 1.0)
        compile_mode: torch.compile mode (default: 'max-autotune')
            Options: None, 'default', 'reduce-overhead', 'max-autotune'

    Attributes:
        tau_m: Memory timescales (property, bounded)
        kappa_m: Memory decay factors = exp(-delta_t / tau_m)
        kappa_b: Branch decay factors = exp(-delta_t / tau_b)
        kappa_lambda: Combined decay = exp(-delta_t * lambda / tau_m)
        w_s: Synapse weights (property, ReLU-activated, LEARNABLE)

    Example:
        >>> model = ELM(
        ...     num_input=784,  # e.g., flattened MNIST patches
        ...     num_output=10,  # 10 classes
        ...     num_memory=100,
        ...     num_branch=196,
        ...     num_synapse_per_branch=4
        ... )
        >>> X = torch.randn(8, 100, 784)  # (batch, time, input)
        >>> Y = model(X)  # (batch, time, 10)
    """

    def __init__(
        self,
        num_input: int,
        num_output: int,
        num_branch: int | None = None,
        num_synapse_per_branch: int = 1,
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
        delta_t: float = 1.0,
        compile_mode: str | None = "max-autotune",
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
        self.delta_t = delta_t
        self.compile_mode = compile_mode

        # Derived properties
        self.mlp_hidden_size = mlp_hidden_size if mlp_hidden_size else 2 * num_memory
        self.num_branch = num_input if num_branch is None else num_branch
        self.num_mlp_input = self.num_branch + num_memory
        self.num_synapse = num_synapse_per_branch * self.num_branch

        # Validate that num_synapse matches num_input
        # (routing is now handled by data transforms)
        assert (
            self.num_synapse == num_input
        ), f"num_synapse ({self.num_synapse}) must equal num_input ({num_input}). Use data transforms for routing."

        # Initialize MLP for nonlinear integration
        # Maps [branch_activations, decayed_previous_memory] to memory_update
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

        # Apply torch.compile to forward and dynamics methods for performance
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode)
            self.dynamics = torch.compile(self.dynamics, mode=compile_mode)

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
        batch_size = x.shape[0]

        # Compute weighted branch input
        # Each branch receives weighted sum of its synapses
        # (w_s * x_t per synapse, then sum per branch)
        b_inp = (w_s * x).view(batch_size, self.num_branch, -1).sum(dim=-1)

        # Update branch activations (exponential filtering)
        # b_t = κ_b * b_{t-1} + branch_input
        b_t = kappa_b * b_prev + b_inp

        # Apply decay to previous memory
        decayed_m_prev = kappa_m * m_prev

        # Compute memory update proposal via MLP
        # Input: [branch_activations, decayed_previous_memory]
        # Output: Δm_t (memory update proposal)
        delta_m_t = custom_tanh(self.mlp(torch.cat([b_t, decayed_m_prev], dim=-1)))

        # Update memory with modified equation (v2 improvement)
        # m_t = κ_m*m_{t-1} + (1 - κ_λ)*Δm_t
        # This replaces v1's λ * (1 - κ_m) formulation
        m_t = decayed_m_prev + (1 - kappa_lambda) * delta_m_t

        # Compute output via linear readout
        y_t = self.w_y(m_t)

        return y_t, b_t, m_t

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
        b_prev = torch.zeros(batch_size, self.num_branch, device=X.device)
        m_prev = torch.zeros(batch_size, self.num_memory, device=X.device)

        # Accumulate outputs
        outputs = []

        # Process each timestep
        for t in range(T):
            y_t, b_prev, m_prev = self.dynamics(
                X[:, t], b_prev, m_prev, w_s, kappa_b, kappa_m, kappa_lambda
            )
            outputs.append(y_t)

        # Stack outputs along time dimension
        return torch.stack(outputs, dim=1)

    def forward_with_states(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns outputs and internal states.

        Useful for visualization, analysis, and debugging of neuron dynamics.
        This is a dataset-agnostic version - task-specific postprocessing
        should be handled by callbacks or wrapper modules.

        Args:
            X: Input sequence (batch, time, num_input)

        Returns:
            tuple of (outputs, branch_record, memory_record):
            - outputs: (batch, time, num_output) raw model predictions
            - branch_record: (batch, time, num_branch) branch activations
            - memory_record: (batch, time, num_memory) memory states
        """
        batch_size, T, _ = X.shape

        # Cache values
        w_s = self.w_s
        kappa_b, kappa_m = self.kappa_b, self.kappa_m
        kappa_lambda = self.kappa_lambda

        # Initialize states
        b_prev = torch.zeros(batch_size, self.num_branch, device=X.device)
        m_prev = torch.zeros(batch_size, self.num_memory, device=X.device)

        # Accumulate outputs and states
        outputs = []
        b_record = []
        m_record = []

        # Process each timestep, recording internal states
        for t in range(T):
            y_t, b_prev, m_prev = self.dynamics(
                X[:, t], b_prev, m_prev, w_s, kappa_b, kappa_m, kappa_lambda
            )
            outputs.append(y_t)
            b_record.append(b_prev)
            m_record.append(m_prev)

        # Stack along time dimension
        outputs = torch.stack(outputs, dim=1)
        b_record = torch.stack(b_record, dim=1)
        m_record = torch.stack(m_record, dim=1)

        return outputs, b_record, m_record
