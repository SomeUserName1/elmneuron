"""
Chaos and Information-Theoretic Analysis of ELM Neurons on NeuronIO Dataset

This script implements methods from methods_summary.md to investigate whether
ELM neurons exhibit chaotic dynamics, particularly near spike initiation.

Usage:
    python analyze_chaos_elm.py --model_path models/elm_dm15.pt --dataset_path /path/to/neuronio

Author: Generated from methods_summary.md
Date: 2025-10-31
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import argparse
from tqdm import tqdm
import h5py

# Import ELM model
from elm_neuron.expressive_leaky_memory_neuron_v2 import ELM


# =============================================================================
# SECTION 1: Data Loading and Model Setup
# =============================================================================


class NeuronIOLoader:
    """Load and preprocess NeuronIO dataset"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_sample(self, sample_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample from NeuronIO dataset

        Returns:
            inputs: Tensor of shape (T, num_input) - synaptic input currents
            targets: Tensor of shape (T, 2) - [spike_target, soma_voltage]
        """
        # TODO: Adapt to actual NeuronIO file structure
        # This is a placeholder - adjust based on actual data format
        test_file = self.dataset_path / "test_data.h5"

        with h5py.File(test_file, "r") as f:
            # Adjust these keys based on actual NeuronIO structure
            inputs = torch.tensor(f["inputs"][sample_idx], dtype=torch.float32)
            spikes = torch.tensor(f["spike_times"][sample_idx], dtype=torch.float32)
            soma = torch.tensor(f["soma_voltage"][sample_idx], dtype=torch.float32)

        targets = torch.stack([spikes, soma], dim=-1)
        return inputs, targets


def load_elm_model(model_path: str, device: str = "cpu") -> ELM:
    """Load pre-trained ELM v2 model"""

    # Initialize model with NeuronIO configuration
    model = ELM(
        num_input=1278,
        num_output=2,
        num_memory=15,  # Adjust based on actual model
        lambda_value=5.0,
        num_branch=45,
        num_synapse_per_branch=100,
        input_to_synapse_routing="neuronio_routing",
        tau_b_value=5.0,
        memory_tau_min=1.0,
        memory_tau_max=150.0,
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    return model


def visualize_data_sample(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
):
    """Visualize a data sample"""

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Input currents
    ax = axes[0]
    im = ax.imshow(
        inputs.T.cpu().numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1
    )
    ax.set_ylabel("Input Channel")
    ax.set_title("Synaptic Input Currents")
    plt.colorbar(im, ax=ax)

    # Ground truth spikes and soma
    ax = axes[1]
    ax.plot(targets[:, 0].cpu().numpy(), label="GT Spikes", color="red")
    ax.plot(targets[:, 1].cpu().numpy(), label="GT Soma V", color="blue", alpha=0.7)
    ax.set_ylabel("Value")
    ax.set_title("Ground Truth")
    ax.legend()

    # Predictions (if provided)
    if predictions is not None:
        ax = axes[2]
        ax.plot(
            predictions[:, 0].cpu().numpy(),
            label="Pred Spikes",
            color="red",
            linestyle="--",
        )
        ax.plot(
            predictions[:, 1].cpu().numpy(),
            label="Pred Soma V",
            color="blue",
            alpha=0.7,
            linestyle="--",
        )
        ax.set_ylabel("Value")
        ax.set_xlabel("Time Step")
        ax.set_title("Model Predictions")
        ax.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 2: Lyapunov Spectrum Computation
# =============================================================================


class LyapunovAnalyzer:
    """Compute Lyapunov spectrum using Jacobian-based QR decomposition method"""

    def __init__(self, model: ELM, device: str = "cpu"):
        self.model = model
        self.device = device

    def compute_jacobian(
        self, state: torch.Tensor, input_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix ∂state_{t+1}/∂state_t

        For ELM v2, state = [b_t, m_t] (branch activations + memory)

        Args:
            state: Current state [b_t, m_t] of shape (num_branch + num_memory,)
            input_t: Input at time t, shape (num_input,)

        Returns:
            J: Jacobian matrix of shape (state_dim, state_dim)
        """
        state = state.clone().detach().requires_grad_(True)

        # Forward pass through one time step
        # This requires modifying model to accept and return internal states
        # For now, we'll use torch.autograd.functional.jacobian

        def state_transition(s):
            # Split state into branches and memory
            num_branches = self.model.num_branch
            b_t = s[:num_branches]
            m_t = s[num_branches:]

            # Compute next state (this is simplified - adapt to actual model)
            # b_{t+1} = κ_b * b_t + weighted_input
            # m_{t+1} = κ_m * m_t + (1 - κ_λ) * Δm_t

            # This requires access to model internals
            # Placeholder - needs actual implementation
            pass

        J = torch.autograd.functional.jacobian(state_transition, state)
        return J

    def compute_spectrum_qr(
        self,
        inputs: torch.Tensor,
        n_exponents: Optional[int] = None,
        t_ons: int = 10,
        warmup: int = 100,
    ) -> np.ndarray:
        """
        Compute full Lyapunov spectrum using QR decomposition

        Args:
            inputs: Input sequence of shape (T, num_input)
            n_exponents: Number of exponents to compute (default: all)
            t_ons: Reorthonormalization frequency
            warmup: Warmup steps before accumulating

        Returns:
            spectrum: Array of Lyapunov exponents sorted descending
        """
        T = len(inputs)
        state_dim = self.model.num_branch + self.model.num_memory

        if n_exponents is None:
            n_exponents = state_dim

        # Initialize perturbation vectors (orthonormal)
        Q = torch.eye(state_dim, n_exponents, device=self.device)
        lyap_sum = torch.zeros(n_exponents, device=self.device)

        # Get initial state
        with torch.no_grad():
            # Run model to get initial state
            # This is placeholder - needs actual state extraction
            state = torch.randn(state_dim, device=self.device)

        n_accumulated = 0

        for t in tqdm(range(T), desc="Computing Lyapunov spectrum"):
            # Compute Jacobian at current state
            J = self.compute_jacobian(state, inputs[t])

            # Propagate perturbations: Q = J @ Q
            Q = J @ Q

            # Reorthonormalization using QR decomposition
            if (t + 1) % t_ons == 0:
                Q, R = torch.linalg.qr(Q)

                if t >= warmup:
                    # Accumulate logarithms of diagonal elements
                    lyap_sum += torch.log(torch.abs(torch.diag(R)))
                    n_accumulated += 1

            # Update state for next iteration
            with torch.no_grad():
                # Placeholder - actual state update
                pass

        # Compute Lyapunov exponents
        spectrum = (lyap_sum / (n_accumulated * t_ons)).cpu().numpy()
        spectrum = np.sort(spectrum)[::-1]  # Sort descending

        return spectrum


def visualize_lyapunov_spectrum(spectrum: np.ndarray, title: str = "Lyapunov Spectrum"):
    """Visualize Lyapunov spectrum"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Full spectrum
    ax = axes[0]
    ax.plot(spectrum, "o-", color="blue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="λ=0")
    ax.set_xlabel("Exponent Index")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Histogram
    ax = axes[1]
    ax.hist(spectrum, bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="λ=0")
    ax.set_xlabel("Lyapunov Exponent Value")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Exponents")
    ax.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 3: Entropy and Dimensionality
# =============================================================================


def kolmogorov_sinai_entropy(lyapunov_spectrum: np.ndarray) -> float:
    """
    Compute Kolmogorov-Sinai entropy rate from Lyapunov spectrum

    H = Σ λ_i for all λ_i > 0

    Args:
        lyapunov_spectrum: Array of Lyapunov exponents

    Returns:
        H: Entropy rate (nats/time)
    """
    positive_exponents = lyapunov_spectrum[lyapunov_spectrum > 0]
    H = np.sum(positive_exponents)
    return H


def kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """
    Compute Kaplan-Yorke (attractor) dimension from Lyapunov spectrum

    D_KY = k + (Σᵢ₌₁ᵏ λ_i) / |λ_{k+1}|

    Args:
        lyapunov_spectrum: Array sorted in descending order

    Returns:
        D_KY: Kaplan-Yorke dimension
    """
    cumsum = np.cumsum(lyapunov_spectrum)
    k = np.where(cumsum >= 0)[0]

    if len(k) == 0:
        return 0.0  # All exponents negative

    k = k[-1]  # Largest index where cumsum >= 0

    if k == len(lyapunov_spectrum) - 1:
        return float(len(lyapunov_spectrum))  # All exponents contribute

    # Kaplan-Yorke formula
    D_KY = k + 1 + cumsum[k] / abs(lyapunov_spectrum[k + 1])
    return D_KY


def visualize_entropy_dimension(spectrum: np.ndarray):
    """Visualize entropy and dimension analysis"""

    H = kolmogorov_sinai_entropy(spectrum)
    D = kaplan_yorke_dimension(spectrum)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Spectrum with positive exponents highlighted
    ax = axes[0]
    positive_mask = spectrum > 0
    ax.plot(
        np.where(positive_mask)[0],
        spectrum[positive_mask],
        "ro",
        label=f"Positive (n={np.sum(positive_mask)})",
        markersize=8,
    )
    ax.plot(
        np.where(~positive_mask)[0],
        spectrum[~positive_mask],
        "bo",
        label="Negative",
        alpha=0.5,
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Exponent Index")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title("Spectrum (Positive Highlighted)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy bar
    ax = axes[1]
    ax.bar(["Entropy Rate"], [H], color="red", alpha=0.7)
    ax.set_ylabel("H (nats/time)")
    ax.set_title(f"Kolmogorov-Sinai Entropy\nH = {H:.4f}")
    ax.grid(True, alpha=0.3, axis="y")

    # Dimension comparison
    ax = axes[2]
    phase_space_dim = len(spectrum)
    ax.bar(
        ["Phase Space\nDim", "Attractor\nDim"],
        [phase_space_dim, D],
        color=["blue", "orange"],
        alpha=0.7,
    )
    ax.set_ylabel("Dimensionality")
    ax.set_title(f"Dimensions\nD_KY = {D:.2f}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, H, D


# =============================================================================
# SECTION 4: Local Analysis (Near Spikes vs Away)
# =============================================================================


def identify_spike_times(targets: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Identify time points near spike initiation"""
    spikes = targets[:, 0].cpu().numpy()
    spike_times = np.where(spikes > threshold)[0]
    return spike_times


def compute_local_lyapunov(
    model: ELM, inputs: torch.Tensor, window_size: int = 50
) -> np.ndarray:
    """
    Compute local maximum Lyapunov exponent in sliding windows

    Args:
        model: ELM model
        inputs: Input sequence
        window_size: Size of sliding window

    Returns:
        local_lambdas: Array of local λ_max values
    """
    T = len(inputs)
    n_windows = T - window_size
    local_lambdas = np.zeros(n_windows)

    analyzer = LyapunovAnalyzer(model)

    for i in tqdm(range(n_windows), desc="Computing local Lyapunov"):
        window_inputs = inputs[i : i + window_size]
        spectrum = analyzer.compute_spectrum_qr(window_inputs, n_exponents=5)
        local_lambdas[i] = spectrum[0]  # Maximum exponent

    return local_lambdas


def visualize_local_chaos(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    local_lambdas: np.ndarray,
    window_size: int = 50,
):
    """Visualize local Lyapunov exponents vs spike times"""

    spike_times = identify_spike_times(targets)
    time_axis = np.arange(len(local_lambdas)) + window_size // 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Input activity
    ax = axes[0]
    ax.plot(
        np.mean(np.abs(inputs.cpu().numpy()), axis=1),
        color="gray",
        alpha=0.5,
        label="Mean |Input|",
    )
    ax.set_ylabel("Mean Input Magnitude")
    ax.set_title("Input Activity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spike times
    ax = axes[1]
    ax.plot(targets[:, 0].cpu().numpy(), color="red", label="Spike Probability")
    for st in spike_times:
        ax.axvline(x=st, color="red", alpha=0.3, linestyle="--")
    ax.set_ylabel("Spike Probability")
    ax.set_title("Spike Initiation Times")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Local Lyapunov exponent
    ax = axes[2]
    ax.plot(time_axis, local_lambdas, color="blue", linewidth=2, label="λ_max (local)")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, label="λ=0")

    # Highlight near-spike regions
    for st in spike_times:
        if st - window_size // 2 >= 0 and st + window_size // 2 < len(local_lambdas):
            idx = st - window_size // 2
            ax.axvline(x=st, color="red", alpha=0.3, linestyle="--")
            ax.plot(time_axis[idx], local_lambdas[idx], "ro", markersize=8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("λ_max")
    ax.set_title("Local Maximum Lyapunov Exponent")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 5: Perturbation Analysis
# =============================================================================


def perturbation_analysis(
    model: ELM, inputs: torch.Tensor, epsilon: float = 1e-3, n_trials: int = 10
) -> Dict[str, np.ndarray]:
    """
    Analyze sensitivity to input perturbations

    Args:
        model: ELM model
        inputs: Input sequence
        epsilon: Perturbation magnitude
        n_trials: Number of perturbation trials

    Returns:
        results: Dict with divergence metrics
    """
    T = len(inputs)

    # Reference trajectory
    with torch.no_grad():
        ref_output = model.neuronio_eval_forward(inputs.unsqueeze(0))[0]

    divergences = np.zeros((n_trials, T))

    for trial in tqdm(range(n_trials), desc="Perturbation analysis"):
        # Add random perturbation
        noise = torch.randn_like(inputs) * epsilon
        perturbed_inputs = inputs + noise

        with torch.no_grad():
            perturbed_output = model.neuronio_eval_forward(
                perturbed_inputs.unsqueeze(0)
            )[0]

        # Compute divergence
        div = torch.norm(perturbed_output - ref_output, dim=-1).cpu().numpy()
        divergences[trial] = div

    return {
        "mean_divergence": np.mean(divergences, axis=0),
        "std_divergence": np.std(divergences, axis=0),
        "max_divergence": np.max(divergences, axis=0),
    }


def visualize_perturbation_analysis(
    results: Dict[str, np.ndarray], targets: torch.Tensor
):
    """Visualize perturbation sensitivity"""

    spike_times = identify_spike_times(targets)
    T = len(results["mean_divergence"])
    time_axis = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Divergence over time
    ax = axes[0]
    ax.fill_between(
        time_axis,
        results["mean_divergence"] - results["std_divergence"],
        results["mean_divergence"] + results["std_divergence"],
        alpha=0.3,
        color="blue",
        label="±1 std",
    )
    ax.plot(
        time_axis,
        results["mean_divergence"],
        color="blue",
        linewidth=2,
        label="Mean divergence",
    )

    # Mark spike times
    for st in spike_times:
        ax.axvline(x=st, color="red", alpha=0.3, linestyle="--")

    ax.set_ylabel("Output Divergence")
    ax.set_title("Sensitivity to Input Perturbations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    ax.semilogy(
        time_axis,
        results["mean_divergence"],
        color="blue",
        linewidth=2,
        label="Mean divergence (log scale)",
    )
    ax.semilogy(
        time_axis,
        results["max_divergence"],
        color="red",
        linewidth=1,
        alpha=0.7,
        label="Max divergence",
    )

    for st in spike_times:
        ax.axvline(x=st, color="red", alpha=0.3, linestyle="--")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Output Divergence (log)")
    ax.set_title("Exponential Divergence Check")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6: Active Information Storage (AIS)
# =============================================================================


def compute_AIS(state_history: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Compute Active Information Storage for each state variable

    AIS measures mutual information I(x^(k)_n; x_{n+1})

    Args:
        state_history: Array of shape (T, d_state)
        k: History length

    Returns:
        AIS: Array of shape (d_state,) - AIS for each state variable
    """
    T, d_state = state_history.shape
    AIS_values = np.zeros(d_state)

    for i in range(d_state):
        x = state_history[:, i]

        # Create history and future
        X_hist = np.array([x[t - k : t] for t in range(k, T - 1)])
        X_future = x[k + 1 : T]

        # Discretize for mutual information estimation
        # Using histogram-based method (simple but may need improvement)
        n_bins = 10

        # 2D histogram for joint distribution
        H_joint, _, _ = np.histogram2d(X_hist.mean(axis=1), X_future, bins=n_bins)
        H_joint = H_joint / H_joint.sum()

        # Marginal distributions
        H_hist = H_joint.sum(axis=1)
        H_fut = H_joint.sum(axis=0)

        # Mutual information
        MI = 0
        for j in range(n_bins):
            for l in range(n_bins):
                if H_joint[j, l] > 0:
                    MI += H_joint[j, l] * np.log(
                        H_joint[j, l] / (H_hist[j] * H_fut[l] + 1e-10)
                    )

        AIS_values[i] = MI

    return AIS_values


def visualize_AIS(AIS_memory: np.ndarray, AIS_branches: np.ndarray):
    """Visualize Active Information Storage"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Memory AIS
    ax = axes[0]
    ax.bar(range(len(AIS_memory)), AIS_memory, color="blue", alpha=0.7)
    ax.set_xlabel("Memory Unit Index")
    ax.set_ylabel("AIS (bits)")
    ax.set_title(
        f"Active Information Storage - Memory\nMean: {np.mean(AIS_memory):.3f}"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Branch AIS
    ax = axes[1]
    ax.bar(range(len(AIS_branches)), AIS_branches, color="green", alpha=0.7)
    ax.set_xlabel("Branch Index")
    ax.set_ylabel("AIS (bits)")
    ax.set_title(
        f"Active Information Storage - Branches\nMean: {np.mean(AIS_branches):.3f}"
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 7: Memory Capacity
# =============================================================================


def compute_memory_capacity(
    model: ELM, input_signal: torch.Tensor, max_delay: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Compute memory capacity through delayed prediction task

    Args:
        model: ELM model
        input_signal: Random input signal
        max_delay: Maximum delay to test

    Returns:
        MC_k: Memory capacity for each delay k
        MC_total: Total memory capacity
    """
    # Generate reservoir states
    with torch.no_grad():
        # Need to extract internal states - requires model modification
        # Placeholder
        reservoir_states = torch.randn(len(input_signal), model.num_memory)

    input_np = input_signal.cpu().numpy()
    states_np = reservoir_states.cpu().numpy()

    MC_k = np.zeros(max_delay)

    for k in tqdm(range(1, max_delay + 1), desc="Computing memory capacity"):
        # Target is input delayed by k steps
        target = input_np[:-k]
        states_delayed = states_np[k:]

        # Train linear readout
        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0)
        reg.fit(states_delayed, target)

        # Predict and compute correlation
        pred = reg.predict(states_delayed)

        # Memory capacity for delay k
        cov = np.cov(target, pred)[0, 1]
        var_target = np.var(target)
        var_pred = np.var(pred)

        MC_k[k - 1] = (cov**2) / (var_target * var_pred + 1e-10)

    MC_total = np.sum(MC_k)
    return MC_k, MC_total


def visualize_memory_capacity(MC_k: np.ndarray, MC_total: float, d_m: int):
    """Visualize memory capacity analysis"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MC as function of delay
    ax = axes[0]
    ax.plot(range(1, len(MC_k) + 1), MC_k, "o-", color="blue")
    ax.set_xlabel("Delay k")
    ax.set_ylabel("MC_k")
    ax.set_title(f"Memory Capacity vs Delay\nTotal MC = {MC_total:.2f}")
    ax.grid(True, alpha=0.3)

    # Cumulative MC
    ax = axes[1]
    ax.plot(range(1, len(MC_k) + 1), np.cumsum(MC_k), color="blue", linewidth=2)
    ax.axhline(y=d_m, color="red", linestyle="--", linewidth=2, label=f"d_m = {d_m}")
    ax.set_xlabel("Delay k")
    ax.set_ylabel("Cumulative MC")
    ax.set_title("Cumulative Memory Capacity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 8: Main Analysis Pipeline
# =============================================================================


def main():
    """Main analysis pipeline"""

    parser = argparse.ArgumentParser(description="Chaos analysis of ELM neurons")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained ELM model"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to NeuronIO dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chaos_analysis_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0, help="Sample index to analyze"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("CHAOS AND INFORMATION-THEORETIC ANALYSIS OF ELM NEURONS")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data and Model
    # =========================================================================
    print("\n[STEP 1] Loading data and model...")

    loader = NeuronIOLoader(args.dataset_path)
    inputs, targets = loader.load_sample(args.sample_idx)
    inputs = inputs.to(args.device)
    targets = targets.to(args.device)

    model = load_elm_model(args.model_path, args.device)

    # Generate predictions
    with torch.no_grad():
        predictions = model.neuronio_eval_forward(inputs.unsqueeze(0))[0]

    # Visualize
    fig = visualize_data_sample(inputs, targets, predictions)
    fig.savefig(output_dir / "step1_data_sample.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved: step1_data_sample.png")
    plt.close()

    # =========================================================================
    # STEP 2: Compute Lyapunov Spectrum
    # =========================================================================
    print("\n[STEP 2] Computing Lyapunov spectrum...")
    print("WARNING: This is computationally intensive and may take several minutes.")
    print("For full validation, consider using a shorter sequence or fewer exponents.")

    analyzer = LyapunovAnalyzer(model, args.device)

    # Compute spectrum (use subset for speed)
    spectrum = analyzer.compute_spectrum_qr(
        inputs[:500],  # Use first 500 steps for speed
        n_exponents=20,  # Compute top 20 exponents
        t_ons=10,
        warmup=50,
    )

    fig = visualize_lyapunov_spectrum(spectrum)
    fig.savefig(
        output_dir / "step2_lyapunov_spectrum.png", dpi=150, bbox_inches="tight"
    )
    print(f"✓ Saved: step2_lyapunov_spectrum.png")
    print(f"  λ_max = {spectrum[0]:.4f}")
    print(f"  # positive exponents = {np.sum(spectrum > 0)}")
    plt.close()

    # =========================================================================
    # STEP 3: Entropy and Dimensionality
    # =========================================================================
    print("\n[STEP 3] Computing entropy and dimensionality...")

    fig, H, D = visualize_entropy_dimension(spectrum)
    fig.savefig(
        output_dir / "step3_entropy_dimension.png", dpi=150, bbox_inches="tight"
    )
    print(f"✓ Saved: step3_entropy_dimension.png")
    print(f"  Entropy rate H = {H:.4f} nats/time")
    print(f"  Attractor dimension D = {D:.2f}")
    print(f"  Phase space dimension = {len(spectrum)}")
    plt.close()

    # =========================================================================
    # STEP 4: Local Analysis (Near Spikes)
    # =========================================================================
    print("\n[STEP 4] Analyzing local dynamics near spikes...")

    local_lambdas = compute_local_lyapunov(model, inputs[:500], window_size=50)

    fig = visualize_local_chaos(inputs[:500], targets[:500], local_lambdas)
    fig.savefig(output_dir / "step4_local_chaos.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved: step4_local_chaos.png")

    spike_times = identify_spike_times(targets[:500])
    if len(spike_times) > 0:
        near_spike_lambdas = local_lambdas[spike_times - 25]  # 25 steps before spike
        away_spike_lambdas = local_lambdas[
            ~np.isin(np.arange(len(local_lambdas)), spike_times)
        ]
        print(
            f"  λ_max near spikes: {np.mean(near_spike_lambdas):.4f} ± {np.std(near_spike_lambdas):.4f}"
        )
        print(
            f"  λ_max away from spikes: {np.mean(away_spike_lambdas):.4f} ± {np.std(away_spike_lambdas):.4f}"
        )
    plt.close()

    # =========================================================================
    # STEP 5: Perturbation Analysis
    # =========================================================================
    print("\n[STEP 5] Analyzing sensitivity to perturbations...")

    results = perturbation_analysis(model, inputs[:500], epsilon=1e-3, n_trials=10)

    fig = visualize_perturbation_analysis(results, targets[:500])
    fig.savefig(output_dir / "step5_perturbations.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved: step5_perturbations.png")
    print(f"  Mean final divergence: {results['mean_divergence'][-1]:.6f}")
    print(f"  Max divergence: {results['max_divergence'].max():.6f}")
    plt.close()

    # =========================================================================
    # STEP 6: Active Information Storage
    # =========================================================================
    print("\n[STEP 6] Computing Active Information Storage...")

    # Extract state history (placeholder - needs actual implementation)
    memory_history = np.random.randn(500, model.num_memory)
    branch_history = np.random.randn(500, model.num_branch)

    AIS_memory = compute_AIS(memory_history, k=2)
    AIS_branches = compute_AIS(branch_history, k=2)

    fig = visualize_AIS(AIS_memory, AIS_branches)
    fig.savefig(output_dir / "step6_AIS.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved: step6_AIS.png")
    print(f"  Mean AIS (memory): {np.mean(AIS_memory):.4f} bits")
    print(f"  Mean AIS (branches): {np.mean(AIS_branches):.4f} bits")
    plt.close()

    # =========================================================================
    # STEP 7: Memory Capacity
    # =========================================================================
    print("\n[STEP 7] Computing memory capacity...")

    # Generate random input signal for MC test
    random_input = torch.randn(1000, device=args.device)

    MC_k, MC_total = compute_memory_capacity(model, random_input, max_delay=50)

    fig = visualize_memory_capacity(MC_k, MC_total, model.num_memory)
    fig.savefig(output_dir / "step7_memory_capacity.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved: step7_memory_capacity.png")
    print(f"  Total memory capacity: {MC_total:.2f}")
    print(f"  Number of memory units: {model.num_memory}")
    print(f"  MC / d_m ratio: {MC_total / model.num_memory:.2f}")
    plt.close()

    # =========================================================================
    # STEP 8: Summary Report
    # =========================================================================
    print("\n[STEP 8] Generating summary report...")

    summary = f"""
CHAOS ANALYSIS SUMMARY
{'='*80}

Model: {args.model_path}
Sample: {args.sample_idx}

LYAPUNOV ANALYSIS:
- Maximum Lyapunov exponent: {spectrum[0]:.4f}
- Number of positive exponents: {np.sum(spectrum > 0)}
- Mean Lyapunov exponent: {np.mean(spectrum):.4f}

ENTROPY AND DIMENSIONALITY:
- Kolmogorov-Sinai entropy rate: {H:.4f} nats/time
- Attractor dimension (Kaplan-Yorke): {D:.2f}
- Phase space dimension: {len(spectrum)}
- Dimensionality ratio D/N: {D/len(spectrum):.3f}

PERTURBATION SENSITIVITY:
- Mean final divergence: {results['mean_divergence'][-1]:.6f}
- Maximum divergence: {results['max_divergence'].max():.6f}

INFORMATION STORAGE:
- Mean AIS (memory): {np.mean(AIS_memory):.4f} bits
- Mean AIS (branches): {np.mean(AIS_branches):.4f} bits

MEMORY CAPACITY:
- Total memory capacity: {MC_total:.2f}
- Number of memory units: {model.num_memory}
- MC / d_m ratio: {MC_total / model.num_memory:.2f}

INTERPRETATION:
"""

    if spectrum[0] > 0:
        summary += "- System exhibits CHAOTIC dynamics (positive λ_max)\n"
    else:
        summary += "- System exhibits STABLE dynamics (negative λ_max)\n"
        summary += "- Check for 'stable chaos' (perturbation sensitivity with negative exponents)\n"

    if H > 0:
        summary += (
            f"- Positive entropy rate indicates sensitivity to initial conditions\n"
        )

    if D < len(spectrum) * 0.5:
        summary += (
            f"- Low-dimensional attractor detected (D << phase space dimension)\n"
        )

    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(summary)

    print(summary)
    print(f"\n✓ Saved: summary_report.txt")
    print(f"\nAll results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
