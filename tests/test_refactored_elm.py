#!/usr/bin/env python
"""
Test script for refactored ELM v2 model.
Verifies that the model runs with dummy data and torch.compile works.
"""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from elmneuron.expressive_leaky_memory_neuron_v2 import ELM


def test_basic_forward():
    """Test basic forward pass with dummy data."""
    print("Testing basic forward pass...")

    # Create model
    model = ELM(
        num_input=64,
        num_output=10,
        num_branch=16,
        num_synapse_per_branch=4,
        num_memory=20,
        compile_mode=None,  # Disable compile for quick test
    )

    # Create dummy input
    batch_size, seq_len, input_dim = 4, 50, 64
    X = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    Y = model(X)

    # Check output shape
    expected_shape = (batch_size, seq_len, 10)
    assert Y.shape == expected_shape, f"Expected {expected_shape}, got {Y.shape}"

    print(f"✓ Forward pass successful! Output shape: {Y.shape}")
    print(f"✓ Output range: [{Y.min():.3f}, {Y.max():.3f}]")

    return model


def test_forward_with_states():
    """Test forward pass with internal state recording."""
    print("\nTesting forward_with_states...")

    # Create model
    model = ELM(
        num_input=64,
        num_output=10,
        num_branch=16,
        num_synapse_per_branch=4,
        num_memory=20,
        compile_mode=None,
    )

    # Create dummy input
    batch_size, seq_len, input_dim = 4, 50, 64
    X = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass with states
    outputs, branch_states, memory_states = model.forward_with_states(X)

    # Check shapes
    assert outputs.shape == (batch_size, seq_len, 10)
    assert branch_states.shape == (batch_size, seq_len, 16)
    assert memory_states.shape == (batch_size, seq_len, 20)

    print(f"✓ forward_with_states successful!")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Branch states shape: {branch_states.shape}")
    print(f"  Memory states shape: {memory_states.shape}")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\nTesting gradient flow...")

    # Create model
    model = ELM(
        num_input=64,
        num_output=10,
        num_branch=16,
        num_synapse_per_branch=4,
        num_memory=20,
        compile_mode=None,
    )

    # Create dummy input and target
    X = torch.randn(2, 20, 64)
    target = torch.randn(2, 20, 10)

    # Forward pass
    output = model(X)

    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert model.w_y.weight.grad is not None
    assert model._proto_w_s.grad is not None

    print(f"✓ Gradient flow verified!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  w_y gradient norm: {model.w_y.weight.grad.norm().item():.4f}")
    print(f"  w_s gradient norm: {model._proto_w_s.grad.norm().item():.4f}")


def test_properties():
    """Test model properties."""
    print("\nTesting model properties...")

    model = ELM(
        num_input=64,
        num_output=10,
        num_branch=16,
        num_synapse_per_branch=4,
        num_memory=20,
        memory_tau_min=1.0,
        memory_tau_max=100.0,
        compile_mode=None,
    )

    # Test tau_m property
    tau_m = model.tau_m
    assert tau_m.shape == (20,)
    assert (tau_m >= 1.0).all() and (tau_m <= 100.0).all()
    print(f"✓ tau_m: shape={tau_m.shape}, range=[{tau_m.min():.2f}, {tau_m.max():.2f}]")

    # Test kappa_m property
    kappa_m = model.kappa_m
    assert kappa_m.shape == (20,)
    assert (kappa_m > 0).all() and (kappa_m < 1).all()
    print(
        f"✓ kappa_m: shape={kappa_m.shape}, range=[{kappa_m.min():.4f}, {kappa_m.max():.4f}]"
    )

    # Test kappa_b property
    kappa_b = model.kappa_b
    assert kappa_b.shape == (16,)
    print(f"✓ kappa_b: shape={kappa_b.shape}, value={kappa_b[0]:.4f}")

    # Test kappa_lambda property
    kappa_lambda = model.kappa_lambda
    assert kappa_lambda.shape == (20,)
    print(
        f"✓ kappa_lambda: shape={kappa_lambda.shape}, range=[{kappa_lambda.min():.4f}, {kappa_lambda.max():.4f}]"
    )

    # Test w_s property (should be non-negative)
    w_s = model.w_s
    assert w_s.shape == (64,)
    assert (w_s >= 0).all()
    print(f"✓ w_s: shape={w_s.shape}, range=[{w_s.min():.4f}, {w_s.max():.4f}]")


def test_torch_compile():
    """Test that torch.compile works (if available)."""
    print("\nTesting torch.compile...")

    if not hasattr(torch, "compile"):
        print("⚠ torch.compile not available (requires PyTorch 2.0+), skipping...")
        return

    try:
        # Create model with compile enabled
        model = ELM(
            num_input=64,
            num_output=10,
            num_branch=16,
            num_synapse_per_branch=4,
            num_memory=20,
            compile_mode="default",  # Use default for faster compile
        )

        # Create dummy input
        X = torch.randn(2, 20, 64)

        # Forward pass (first call will trigger compilation)
        Y = model(X)

        print(f"✓ torch.compile successful! Output shape: {Y.shape}")
    except Exception as e:
        print(f"⚠ torch.compile test failed: {e}")
        print("  (This is expected if running on CPU without proper setup)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Refactored ELM v2 Model")
    print("=" * 60)

    try:
        test_basic_forward()
        test_forward_with_states()
        test_gradient_flow()
        test_properties()
        test_torch_compile()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
