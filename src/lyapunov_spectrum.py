"""
Lyapunov Spectrum Calculation for ELM Neuron Models

This module implements methods to compute the Lyapunov spectrum for the
Expressive Leaky Memory (ELM) neuron model, which characterizes the
sensitivity of the dynamical system to perturbations in different directions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm


class LyapunovSpectrumCalculator:
    """
    Calculate Lyapunov spectrum for ELM neuron using QR decomposition method.

    The Lyapunov spectrum quantifies the average exponential rates of divergence
    (positive exponents) or convergence (negative exponents) of nearby trajectories
    in state space.
    """

    def __init__(
        self,
        model: nn.Module,
        model_version: str = "v1",  # "v1" or "v2"
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Lyapunov spectrum calculator.

        Args:
            model: ELM model instance
            model_version: Which version of ELM ("v1" uses synapse states, "v2" uses branch states)
            device: Torch device to use for computation
        """
        self.model = model
        self.model_version = model_version
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()

        # Determine state dimensions based on model version
        if model_version == "v1":
            self.intermediate_dim = model.num_synapse
        elif model_version == "v2":
            self.intermediate_dim = model.num_branch
        else:
            raise ValueError(f"Unknown model version: {model_version}")

        self.memory_dim = model.num_memory
        self.state_dim = self.intermediate_dim + self.memory_dim

    def compute_jacobian(
        self,
        x_t: torch.Tensor,
        state_prev: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Jacobian of the dynamics with respect to the state.

        Args:
            x_t: Input at time t, shape (batch_size, num_input)
            state_prev: Tuple of (intermediate_state, memory_state) from previous timestep
                - intermediate_state: shape (batch_size, intermediate_dim)
                - memory_state: shape (batch_size, memory_dim)

        Returns:
            Jacobian tensor of shape (batch_size, state_dim, state_dim)
        """
        intermediate_prev, m_prev = state_prev
        batch_size = x_t.shape[0]

        # Concatenate states for gradient computation
        state_vec = torch.cat([intermediate_prev, m_prev], dim=-1)
        state_vec.requires_grad_(True)

        # Split state back
        intermediate = state_vec[:, :self.intermediate_dim]
        m = state_vec[:, self.intermediate_dim:]

        # Compute next state using model dynamics
        with torch.enable_grad():
            if self.model_version == "v1":
                # v1: synapse dynamics
                s_t = self.model.kappa_s * intermediate + self.model.w_s * x_t
                syn_input = s_t.view(batch_size, self.model.num_branch, -1).sum(dim=-1)
                delta_m_t = torch.tanh(
                    self.model.mlp(torch.cat([syn_input, self.model.kappa_m * m], dim=-1))
                )
                m_t = self.model.kappa_m * m + self.model.lambda_value * (1 - self.model.kappa_m) * delta_m_t
                next_state = torch.cat([s_t, m_t], dim=-1)
            else:
                # v2: branch dynamics
                b_inp = (self.model.w_s * x_t).view(batch_size, self.model.num_branch, -1).sum(dim=-1)
                b_t = self.model.kappa_b * intermediate + b_inp
                delta_m_t = torch.tanh(
                    self.model.mlp(torch.cat([b_t, self.model.kappa_m * m], dim=-1))
                )
                m_t = self.model.kappa_m * m + (1 - self.model.kappa_lambda) * delta_m_t
                next_state = torch.cat([b_t, m_t], dim=-1)

        # Compute Jacobian using autograd
        jacobian = torch.zeros(batch_size, self.state_dim, self.state_dim, device=self.device)
        for i in range(self.state_dim):
            grad_outputs = torch.zeros_like(next_state)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(
                outputs=next_state,
                inputs=state_vec,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )[0]
            jacobian[:, i, :] = grads

        return jacobian

    def step_with_jacobian(
        self,
        x_t: torch.Tensor,
        state_prev: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Perform one step of dynamics and compute the Jacobian.

        Args:
            x_t: Input at time t
            state_prev: Previous state tuple

        Returns:
            Tuple of (next_state, jacobian)
        """
        batch_size = x_t.shape[0]
        intermediate_prev, m_prev = state_prev

        # Forward dynamics (no grad needed for state evolution)
        with torch.no_grad():
            if self.model_version == "v1":
                w_s, kappa_s, kappa_m = self.model.w_s, self.model.kappa_s, self.model.kappa_m
                s_t = kappa_s * intermediate_prev + w_s * x_t
                syn_input = s_t.view(batch_size, self.model.num_branch, -1).sum(dim=-1)
                delta_m_t = torch.tanh(
                    self.model.mlp(torch.cat([syn_input, kappa_m * m_prev], dim=-1))
                )
                m_t = kappa_m * m_prev + self.model.lambda_value * (1 - kappa_m) * delta_m_t
                next_state = (s_t, m_t)
            else:
                w_s = self.model.w_s
                kappa_b, kappa_m, kappa_lambda = self.model.kappa_b, self.model.kappa_m, self.model.kappa_lambda
                b_inp = (w_s * x_t).view(batch_size, self.model.num_branch, -1).sum(dim=-1)
                b_t = kappa_b * intermediate_prev + b_inp
                delta_m_t = torch.tanh(
                    self.model.mlp(torch.cat([b_t, kappa_m * m_prev], dim=-1))
                )
                m_t = kappa_m * m_prev + (1 - kappa_lambda) * delta_m_t
                next_state = (b_t, m_t)

        # Compute Jacobian
        jacobian = self.compute_jacobian(x_t, state_prev)

        return next_state, jacobian

    def compute_spectrum(
        self,
        inputs: torch.Tensor,
        num_transient: int = 500,
        num_iterations: int = 5000,
        orthonormalize_every: int = 1,
        return_trajectory: bool = False,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the Lyapunov spectrum using the QR decomposition method.

        Args:
            inputs: Input sequence, shape (batch_size, T, num_input)
            num_transient: Number of initial timesteps to discard (transient dynamics)
            num_iterations: Number of timesteps to use for Lyapunov computation
            orthonormalize_every: Frequency of QR orthonormalization
            return_trajectory: Whether to return the state trajectory
            verbose: Whether to show progress bar

        Returns:
            Dictionary containing:
                - 'lyapunov_exponents': Array of shape (batch_size, state_dim)
                - 'trajectory': Optional, state trajectory if return_trajectory=True
        """
        batch_size, T, _ = inputs.shape

        if T < num_transient + num_iterations:
            raise ValueError(
                f"Input sequence length ({T}) must be >= num_transient + num_iterations "
                f"({num_transient + num_iterations})"
            )

        # Initialize state
        intermediate = torch.zeros(batch_size, self.intermediate_dim, device=self.device)
        m = torch.zeros(batch_size, self.memory_dim, device=self.device)
        state = (intermediate, m)

        # Initialize orthonormal tangent vectors using QR decomposition
        Q = torch.zeros(batch_size, self.state_dim, self.state_dim, device=self.device)
        for b in range(batch_size):
            Q[b] = torch.linalg.qr(torch.randn(self.state_dim, self.state_dim, device=self.device))[0]

        # Accumulate log of stretching factors
        lyapunov_sum = torch.zeros(batch_size, self.state_dim, device=self.device)

        # Optional trajectory storage
        trajectory = [] if return_trajectory else None

        # Route inputs to synapses
        inputs_routed = self.model.route_input_to_synapses(inputs)

        # Transient period - let the system settle
        if verbose:
            print("Running transient period...")
        for t in range(num_transient):
            with torch.no_grad():
                if self.model_version == "v1":
                    w_s, kappa_s, kappa_m = self.model.w_s, self.model.kappa_s, self.model.kappa_m
                    s_t = kappa_s * state[0] + w_s * inputs_routed[:, t]
                    syn_input = s_t.view(batch_size, self.model.num_branch, -1).sum(dim=-1)
                    delta_m_t = torch.tanh(
                        self.model.mlp(torch.cat([syn_input, kappa_m * state[1]], dim=-1))
                    )
                    m_t = kappa_m * state[1] + self.model.lambda_value * (1 - kappa_m) * delta_m_t
                    state = (s_t, m_t)
                else:
                    w_s = self.model.w_s
                    kappa_b, kappa_m, kappa_lambda = self.model.kappa_b, self.model.kappa_m, self.model.kappa_lambda
                    b_inp = (w_s * inputs_routed[:, t]).view(batch_size, self.model.num_branch, -1).sum(dim=-1)
                    b_t = kappa_b * state[0] + b_inp
                    delta_m_t = torch.tanh(
                        self.model.mlp(torch.cat([b_t, kappa_m * state[1]], dim=-1))
                    )
                    m_t = kappa_m * state[1] + (1 - kappa_lambda) * delta_m_t
                    state = (b_t, m_t)

        # Main computation loop
        iterator = range(num_transient, num_transient + num_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Computing Lyapunov spectrum")

        for t in iterator:
            x_t = inputs_routed[:, t]

            # Evolve state and get Jacobian
            state, J = self.step_with_jacobian(x_t, state)

            if return_trajectory:
                trajectory.append(torch.cat([state[0], state[1]], dim=-1))

            # Evolve tangent vectors: Q_new = J @ Q
            Q = torch.bmm(J, Q)

            # Orthonormalize periodically
            if (t - num_transient) % orthonormalize_every == 0:
                for b in range(batch_size):
                    Q_b, R_b = torch.linalg.qr(Q[b])
                    Q[b] = Q_b
                    # Accumulate log of diagonal elements (stretching factors)
                    lyapunov_sum[b] += torch.log(torch.abs(torch.diag(R_b)))

        # Final orthonormalization if needed
        if (num_iterations - 1) % orthonormalize_every != 0:
            for b in range(batch_size):
                Q_b, R_b = torch.linalg.qr(Q[b])
                lyapunov_sum[b] += torch.log(torch.abs(torch.diag(R_b)))

        # Compute Lyapunov exponents (average growth rate per timestep)
        lyapunov_exponents = lyapunov_sum / num_iterations

        # Convert to numpy for easier analysis
        result = {
            'lyapunov_exponents': lyapunov_exponents.cpu().numpy(),
            'positive_exponents': (lyapunov_exponents > 0).sum(dim=-1).cpu().numpy(),
            'max_exponent': lyapunov_exponents.max(dim=-1)[0].cpu().numpy(),
            'sum_positive': torch.clamp(lyapunov_exponents, min=0).sum(dim=-1).cpu().numpy(),
        }

        if return_trajectory:
            result['trajectory'] = torch.stack(trajectory, dim=1).cpu().numpy()

        return result

    def compute_spectrum_from_data_loader(
        self,
        data_loader,
        num_samples: int = 10,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute Lyapunov spectrum from a data loader.

        Args:
            data_loader: Data loader providing input sequences
            num_samples: Number of samples to process
            **kwargs: Additional arguments passed to compute_spectrum

        Returns:
            Dictionary with aggregated results across all samples
        """
        all_exponents = []
        all_max_exponents = []
        all_positive_counts = []

        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_samples:
                break

            inputs = inputs.to(self.device)
            result = self.compute_spectrum(inputs, verbose=False, **kwargs)

            all_exponents.append(result['lyapunov_exponents'])
            all_max_exponents.append(result['max_exponent'])
            all_positive_counts.append(result['positive_exponents'])

        # Aggregate results
        all_exponents = np.concatenate(all_exponents, axis=0)

        return {
            'lyapunov_exponents': all_exponents,
            'mean_spectrum': np.mean(all_exponents, axis=0),
            'std_spectrum': np.std(all_exponents, axis=0),
            'max_exponent_mean': np.mean(all_max_exponents),
            'max_exponent_std': np.std(all_max_exponents),
            'positive_count_mean': np.mean(all_positive_counts),
            'positive_count_std': np.std(all_positive_counts),
        }


def analyze_lyapunov_spectrum(
    spectrum_results: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze Lyapunov spectrum results and extract key metrics.

    Args:
        spectrum_results: Results from compute_spectrum or compute_spectrum_from_data_loader
        verbose: Whether to print analysis

    Returns:
        Dictionary with analysis metrics
    """
    if 'mean_spectrum' in spectrum_results:
        # Results from data loader
        spectrum = spectrum_results['mean_spectrum']
    else:
        # Results from single computation
        spectrum = np.mean(spectrum_results['lyapunov_exponents'], axis=0)

    # Sort in descending order
    spectrum_sorted = np.sort(spectrum)[::-1]

    # Key metrics
    max_exponent = spectrum_sorted[0]
    num_positive = np.sum(spectrum > 0)
    num_zero = np.sum(np.abs(spectrum) < 1e-6)
    num_negative = np.sum(spectrum < -1e-6)
    lyapunov_dimension = np.sum(spectrum > 0)  # Simplified Kaplan-Yorke dimension
    sum_positive = np.sum(spectrum[spectrum > 0])

    metrics = {
        'max_lyapunov_exponent': float(max_exponent),
        'num_positive_exponents': int(num_positive),
        'num_zero_exponents': int(num_zero),
        'num_negative_exponents': int(num_negative),
        'lyapunov_dimension': float(lyapunov_dimension),
        'sum_positive_exponents': float(sum_positive),
        'mean_exponent': float(np.mean(spectrum)),
    }

    if verbose:
        print("\n" + "="*60)
        print("Lyapunov Spectrum Analysis")
        print("="*60)
        print(f"State dimension: {len(spectrum)}")
        print(f"Max Lyapunov exponent: {max_exponent:.6f}")
        print(f"Number of positive exponents: {num_positive}")
        print(f"Number of zero exponents: {num_zero}")
        print(f"Number of negative exponents: {num_negative}")
        print(f"Lyapunov dimension: {lyapunov_dimension:.2f}")
        print(f"Sum of positive exponents: {sum_positive:.6f}")
        print(f"Mean exponent: {np.mean(spectrum):.6f}")
        print("\nSpectrum interpretation:")
        if max_exponent > 0.01:
            print("  - System exhibits CHAOTIC behavior (positive max exponent)")
        elif max_exponent > -0.01:
            print("  - System is on the EDGE OF CHAOS (near-zero max exponent)")
        else:
            print("  - System exhibits STABLE/PERIODIC behavior (negative max exponent)")
        print("="*60 + "\n")

    return metrics
