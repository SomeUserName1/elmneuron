"""
Test chaos analysis methods on synthetic data

This script tests the core methods from analyze_chaos_elm.py using
simple synthetic systems with known properties.

Usage:
    python test_chaos_methods.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm

# Import methods from analyze_chaos_elm
from analyze_chaos_elm import (
    kolmogorov_sinai_entropy,
    kaplan_yorke_dimension,
    compute_AIS,
    visualize_lyapunov_spectrum,
    visualize_entropy_dimension,
)


def lorenz_system(state, sigma=10.0, rho=28.0, beta=8 / 3):
    """Lorenz system: classic chaotic system for testing"""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def lorenz_jacobian(state, sigma=10.0, rho=28.0, beta=8 / 3):
    """Jacobian of Lorenz system"""
    x, y, z = state
    J = np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])
    return J


def compute_lyapunov_lorenz(n_steps=10000, dt=0.01, t_ons=10):
    """
    Compute Lyapunov spectrum for Lorenz system
    Known values: λ ≈ [0.9, 0, -14.5]
    """
    # Initial condition
    state = np.array([1.0, 1.0, 1.0])

    # Initialize perturbation vectors
    Q = np.eye(3)
    lyap_sum = np.zeros(3)
    n_accumulated = 0

    print("Computing Lyapunov spectrum for Lorenz system...")
    print(f"Known approximate values: λ ≈ [0.9, 0, -14.5]")

    for step in range(n_steps):
        # Runge-Kutta 4th order integration
        k1 = lorenz_system(state)
        k2 = lorenz_system(state + 0.5 * dt * k1)
        k3 = lorenz_system(state + 0.5 * dt * k2)
        k4 = lorenz_system(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Propagate perturbations
        J = lorenz_jacobian(state)

        # Linearized flow: Φ = exp(J*dt) using matrix exponential
        # This is more accurate than first-order Euler approximation
        Phi = expm(J * dt)
        Q = Phi @ Q

        # Reorthonormalization
        if (step + 1) % t_ons == 0:
            Q, R = np.linalg.qr(Q)

            if step > 1000:  # Warmup
                lyap_sum += np.log(np.abs(np.diag(R)))
                n_accumulated += 1

    # Compute Lyapunov exponents (convert to per-unit-time)
    spectrum = lyap_sum / (n_accumulated * t_ons * dt)
    return spectrum


def henon_map(state, a=1.4, b=0.3):
    """
    Henon map: discrete-time chaotic system
    Known: λ ≈ [0.42, -1.62]
    """
    x, y = state
    x_next = 1 - a * x**2 + y
    y_next = b * x
    return np.array([x_next, y_next])


def henon_jacobian(state, a=1.4, b=0.3):
    """Jacobian of Henon map"""
    x, y = state
    J = np.array([[-2 * a * x, 1], [b, 0]])
    return J


def compute_lyapunov_henon(n_steps=10000, t_ons=10):
    """
    Compute Lyapunov spectrum for Henon map
    Known values: λ ≈ [0.42, -1.62]
    """
    state = np.array([0.1, 0.1])

    Q = np.eye(2)
    lyap_sum = np.zeros(2)
    n_accumulated = 0

    print("Computing Lyapunov spectrum for Henon map...")
    print(f"Known approximate values: λ ≈ [0.42, -1.62]")

    for step in range(n_steps):
        state = henon_map(state)
        J = henon_jacobian(state)

        Q = J @ Q

        if (step + 1) % t_ons == 0:
            Q, R = np.linalg.qr(Q)

            if step > 1000:
                lyap_sum += np.log(np.abs(np.diag(R)))
                n_accumulated += 1

    spectrum = lyap_sum / (n_accumulated * t_ons)
    return spectrum


def test_entropy_dimension():
    """Test entropy and dimension calculations"""

    print("\n" + "=" * 80)
    print("TEST 1: Entropy and Dimension Calculations")
    print("=" * 80)

    # Test case 1: All negative exponents (stable)
    spectrum1 = np.array([-0.1, -0.5, -1.0, -2.0, -3.0])
    H1 = kolmogorov_sinai_entropy(spectrum1)
    D1 = kaplan_yorke_dimension(spectrum1)

    print("\nCase 1: All negative exponents (stable system)")
    print(f"Spectrum: {spectrum1}")
    print(f"Entropy H: {H1:.4f} (expected: 0)")
    print(f"Dimension D: {D1:.4f} (expected: 0)")
    assert H1 == 0, "Entropy should be 0 for stable system"
    assert D1 == 0, "Dimension should be 0 for stable system"
    print("✓ PASSED")

    # Test case 2: One positive exponent (chaotic)
    spectrum2 = np.array([0.5, -0.1, -0.5, -1.0, -2.0])
    H2 = kolmogorov_sinai_entropy(spectrum2)
    D2 = kaplan_yorke_dimension(spectrum2)

    print("\nCase 2: One positive exponent (chaotic system)")
    print(f"Spectrum: {spectrum2}")
    print(f"Entropy H: {H2:.4f} (expected: 0.5)")
    print(f"Dimension D: {D2:.4f}")
    assert abs(H2 - 0.5) < 1e-6, "Entropy should equal positive exponent"
    print("✓ PASSED")

    # Test case 3: Multiple positive exponents
    spectrum3 = np.array([1.0, 0.5, 0.2, -0.5, -1.0])
    H3 = kolmogorov_sinai_entropy(spectrum3)
    D3 = kaplan_yorke_dimension(spectrum3)

    print("\nCase 3: Multiple positive exponents")
    print(f"Spectrum: {spectrum3}")
    print(f"Entropy H: {H3:.4f} (expected: 1.7)")
    print(f"Dimension D: {D3:.4f}")
    expected_H = 1.0 + 0.5 + 0.2
    assert abs(H3 - expected_H) < 1e-6, "Entropy should equal sum of positive exponents"
    print("✓ PASSED")

    # Visualize
    fig = visualize_lyapunov_spectrum(spectrum3, "Test Case 3")
    fig.savefig("test_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, H, D = visualize_entropy_dimension(spectrum3)
    fig.savefig("test_entropy_dimension.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n✓ All entropy/dimension tests PASSED")
    print("✓ Saved: test_spectrum.png, test_entropy_dimension.png")


def test_chaotic_systems():
    """Test on known chaotic systems"""

    print("\n" + "=" * 80)
    print("TEST 2: Known Chaotic Systems")
    print("=" * 80)

    # Test Henon map
    print("\n--- Henon Map ---")
    spectrum_henon = compute_lyapunov_henon(n_steps=10000)
    H_henon = kolmogorov_sinai_entropy(spectrum_henon)
    D_henon = kaplan_yorke_dimension(spectrum_henon)

    print(f"\nResults:")
    print(f"  λ₁ = {spectrum_henon[0]:.4f} (expected: ~0.42)")
    print(f"  λ₂ = {spectrum_henon[1]:.4f} (expected: ~-1.62)")
    print(f"  H = {H_henon:.4f}")
    print(f"  D = {D_henon:.4f}")

    # Check approximate match (allow 10% error)
    assert abs(spectrum_henon[0] - 0.42) < 0.05, "Henon λ₁ should be ~0.42"
    assert abs(spectrum_henon[1] - (-1.62)) < 0.2, "Henon λ₂ should be ~-1.62"
    print("✓ PASSED (within tolerance)")

    # Test Lorenz system
    print("\n--- Lorenz System ---")
    spectrum_lorenz = compute_lyapunov_lorenz(n_steps=10000, dt=0.01)
    H_lorenz = kolmogorov_sinai_entropy(spectrum_lorenz)
    D_lorenz = kaplan_yorke_dimension(spectrum_lorenz)

    print(f"\nResults:")
    print(f"  λ₁ = {spectrum_lorenz[0]:.4f} (expected: ~0.9)")
    print(f"  λ₂ = {spectrum_lorenz[1]:.4f} (expected: ~0)")
    print(f"  λ₃ = {spectrum_lorenz[2]:.4f} (expected: ~-14.5)")
    print(f"  H = {H_lorenz:.4f}")
    print(f"  D = {D_lorenz:.4f} (expected: ~2.06)")

    # Check approximate match
    assert abs(spectrum_lorenz[0] - 0.9) < 0.2, "Lorenz λ₁ should be ~0.9"
    assert abs(spectrum_lorenz[1]) < 0.1, "Lorenz λ₂ should be ~0"
    assert abs(spectrum_lorenz[2] - (-14.5)) < 2.0, "Lorenz λ₃ should be ~-14.5"
    print("✓ PASSED (within tolerance)")

    # Visualize both
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Henon spectrum
    ax = axes[0, 0]
    ax.plot(spectrum_henon, "o-", color="blue")
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_title("Henon Map Spectrum")
    ax.set_ylabel("Lyapunov Exponent")
    ax.grid(True, alpha=0.3)

    # Lorenz spectrum
    ax = axes[0, 1]
    ax.plot(spectrum_lorenz, "o-", color="green")
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_title("Lorenz System Spectrum")
    ax.set_ylabel("Lyapunov Exponent")
    ax.grid(True, alpha=0.3)

    # Comparison table
    ax = axes[1, 0]
    ax.axis("off")
    table_data = [
        ["System", "λ_max", "H", "D"],
        ["Henon", f"{spectrum_henon[0]:.3f}", f"{H_henon:.3f}", f"{D_henon:.3f}"],
        ["Lorenz", f"{spectrum_lorenz[0]:.3f}", f"{H_lorenz:.3f}", f"{D_lorenz:.3f}"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title("Comparison")

    # Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"Henon Map:\n"
        f"  Known: λ ≈ [0.42, -1.62]\n"
        f"  Computed: [{spectrum_henon[0]:.3f}, {spectrum_henon[1]:.3f}]\n"
        f"  Status: {'✓ PASS' if abs(spectrum_henon[0]-0.42)<0.1 else '✗ FAIL'}\n\n"
        f"Lorenz System:\n"
        f"  Known: λ ≈ [0.9, 0, -14.5]\n"
        f"  Computed: [{spectrum_lorenz[0]:.3f}, {spectrum_lorenz[1]:.3f}, {spectrum_lorenz[2]:.3f}]\n"
        f"  Status: {'✓ PASS' if abs(spectrum_lorenz[0]-0.9)<0.2 else '✗ FAIL'}"
    )
    ax.text(
        0.1, 0.5, summary, fontsize=10, family="monospace", verticalalignment="center"
    )

    plt.tight_layout()
    fig.savefig("test_chaotic_systems.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n✓ Chaotic systems tests PASSED")
    print("✓ Saved: test_chaotic_systems.png")


def test_AIS():
    """Test Active Information Storage computation"""

    print("\n" + "=" * 80)
    print("TEST 3: Active Information Storage")
    print("=" * 80)

    # Generate AR(1) process: x_t = 0.8 * x_{t-1} + noise
    # Should have high AIS due to strong autocorrelation
    np.random.seed(42)
    T = 1000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.8 * x[t - 1] + np.random.randn() * 0.1

    # Compute AIS
    state_history = x.reshape(-1, 1)
    AIS = compute_AIS(state_history, k=2)

    print(f"\nAR(1) process (ρ=0.8):")
    print(f"  AIS = {AIS[0]:.4f} bits")
    print(f"  Expected: Moderate-to-high AIS due to autocorrelation")

    # Random walk should have lower AIS
    y = np.cumsum(np.random.randn(T))
    state_history_rw = y.reshape(-1, 1)
    AIS_rw = compute_AIS(state_history_rw, k=2)

    print(f"\nRandom walk:")
    print(f"  AIS = {AIS_rw[0]:.4f} bits")

    # White noise should have minimal AIS
    z = np.random.randn(T)
    state_history_wn = z.reshape(-1, 1)
    AIS_wn = compute_AIS(state_history_wn, k=2)

    print(f"\nWhite noise:")
    print(f"  AIS = {AIS_wn[0]:.4f} bits")
    print(f"  Expected: Near zero (no temporal structure)")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series
    ax = axes[0, 0]
    ax.plot(x[:200], label="AR(1)", alpha=0.7)
    ax.plot(y[:200], label="Random Walk", alpha=0.7)
    ax.plot(z[:200], label="White Noise", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Test Time Series")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AIS comparison
    ax = axes[0, 1]
    ax.bar(
        ["AR(1)", "Random Walk", "White Noise"],
        [AIS[0], AIS_rw[0], AIS_wn[0]],
        color=["blue", "orange", "green"],
        alpha=0.7,
    )
    ax.set_ylabel("AIS (bits)")
    ax.set_title("Active Information Storage")
    ax.grid(True, alpha=0.3, axis="y")

    # Autocorrelation
    ax = axes[1, 0]
    ax.acorr(x, maxlags=50, usevlines=True, label="AR(1)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation (AR1)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.acorr(z, maxlags=50, usevlines=True, label="White Noise")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation (White Noise)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("test_AIS.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n✓ AIS test completed")
    print("✓ Saved: test_AIS.png")
    print("\nNote: AIS values depend on discretization. Check visually that:")
    print("  - AR(1) has highest AIS (strongest temporal structure)")
    print("  - White noise has lowest AIS (no temporal structure)")


def main():
    """Run all tests"""

    print("\n" + "=" * 80)
    print("TESTING CHAOS ANALYSIS METHODS")
    print("=" * 80)
    print("\nThis script validates core methods using known chaotic systems.")
    print("Results will be saved as test_*.png files.\n")

    # Run tests
    test_entropy_dimension()
    test_chaotic_systems()
    test_AIS()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - test_spectrum.png")
    print("  - test_entropy_dimension.png")
    print("  - test_chaotic_systems.png")
    print("  - test_AIS.png")
    print("\nNext steps:")
    print("  1. Review the generated plots")
    print("  2. Verify results match known values")
    print("  3. If tests pass, proceed to analyze_chaos_elm.py")
    print("  4. Complete the Jacobian and state extraction implementations")
    print("\n")


if __name__ == "__main__":
    main()
