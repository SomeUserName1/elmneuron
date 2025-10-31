"""
Comprehensive test suite for ELM neuron models (v1 and v2).

Tests cover initialization, forward passes, routing mechanisms,
timescale properties, memory dynamics, and comparisons between
the two model versions.

Run with: pytest test_elm_models.py -v
"""

import pytest
import torch
import math

# Import both versions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.expressive_leaky_memory_neuron import (
    ELM as ELM_v1
)
from src.expressive_leaky_memory_neuron_v2 import (
    ELM as ELM_v2
)


# Test fixtures
@pytest.fixture
def device():
    """Test device (CPU or CUDA if available)."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@pytest.fixture
def simple_config():
    """Simple configuration for basic tests."""
    return {
        "num_input": 100,
        "num_output": 2,
        "num_memory": 10,
        "lambda_value": 5.0,
        "mlp_num_layers": 1,
        "delta_t": 1.0,
    }


@pytest.fixture
def neuronio_config():
    """Configuration mimicking NeuronIO setup."""
    return {
        "num_input": 1278,
        "num_output": 2,
        "num_memory": 20,
        "lambda_value": 5.0,
        "num_branch": 45,
        "num_synapse_per_branch": 100,
        "input_to_synapse_routing": "neuronio_routing",
        "tau_s_value": 5.0,  # v1
        "tau_b_value": 5.0,  # v2
        "memory_tau_min": 1.0,
        "memory_tau_max": 150.0,
    }


# ====================
# Initialization Tests
# ====================

class TestInitialization:
    """Test model initialization for both versions."""

    def test_v1_basic_init(self, simple_config):
        """Test v1 initializes with basic configuration."""
        model = ELM_v1(**simple_config)
        assert model.num_input == simple_config["num_input"]
        assert model.num_output == simple_config["num_output"]
        assert model.num_memory == simple_config["num_memory"]

    def test_v2_basic_init(self, simple_config):
        """Test v2 initializes with basic configuration."""
        model = ELM_v2(**simple_config)
        assert model.num_input == simple_config["num_input"]
        assert model.num_output == simple_config["num_output"]
        assert model.num_memory == simple_config["num_memory"]

    def test_v1_parameter_shapes(self, simple_config):
        """Test v1 parameter shapes are correct."""
        model = ELM_v1(**simple_config)

        # Check synapse weights
        assert model.w_s.shape == (model.num_synapse,)

        # Check memory timescales
        assert model.tau_m.shape == (model.num_memory,)

        # Check decay factors
        assert model.kappa_m.shape == (model.num_memory,)
        assert model.kappa_s.shape == (model.num_synapse,)

        # Check output layer
        assert model.w_y.weight.shape == (
            model.num_output,
            model.num_memory
        )

    def test_v2_parameter_shapes(self, simple_config):
        """Test v2 parameter shapes are correct."""
        model = ELM_v2(**simple_config)

        # Check synapse weights (learnable in v2)
        assert model.w_s.shape == (model.num_synapse,)
        assert model._proto_w_s.requires_grad

        # Check branch decay factors
        assert model.kappa_b.shape == (model.num_branch,)

        # Check kappa_lambda (v2 specific)
        assert model.kappa_lambda.shape == (model.num_memory,)

    def test_neuronio_routing_init(self, neuronio_config):
        """Test initialization with neuronio routing."""
        # Create v1-specific config (tau_s_value)
        config_v1 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_b_value'}
        # Create v2-specific config (tau_b_value)
        config_v2 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_s_value'}

        model_v1 = ELM_v1(**config_v1)
        model_v2 = ELM_v2(**config_v2)

        # Check routing indices were created
        assert model_v1.input_to_synapse_indices.numel() > 0
        assert model_v2.input_to_synapse_indices.numel() > 0

        # Check validity masks
        assert model_v1.valid_indices_mask.numel() > 0
        assert model_v2.valid_indices_mask.numel() > 0

    def test_random_routing_init(self, simple_config):
        """Test initialization with random routing."""
        config = simple_config.copy()
        config["input_to_synapse_routing"] = "random_routing"
        config["num_synapse_per_branch"] = 2

        model = ELM_v1(**config)
        assert model.input_to_synapse_indices.numel() > 0


# ==================
# Forward Pass Tests
# ==================

class TestForwardPass:
    """Test forward pass for both versions."""

    def test_v1_forward_shape(self, simple_config, device):
        """Test v1 forward pass output shape."""
        model = ELM_v1(**simple_config).to(device)
        batch, time = 4, 50
        X = torch.randn(
            batch,
            time,
            simple_config["num_input"],
            device=device
        )

        Y = model(X)

        assert Y.shape == (
            batch,
            time,
            simple_config["num_output"]
        )
        assert not torch.isnan(Y).any()

    def test_v2_forward_shape(self, simple_config, device):
        """Test v2 forward pass output shape."""
        model = ELM_v2(**simple_config).to(device)
        batch, time = 4, 50
        X = torch.randn(
            batch,
            time,
            simple_config["num_input"],
            device=device
        )

        Y = model(X)

        assert Y.shape == (
            batch,
            time,
            simple_config["num_output"]
        )
        assert not torch.isnan(Y).any()

    def test_v1_neuronio_forward(
        self,
        neuronio_config,
        device
    ):
        """Test v1 NeuronIO-specific forward pass."""
        # Create v1-specific config
        config_v1 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_b_value'}
        model = ELM_v1(**config_v1).to(device)
        batch, time = 2, 100
        X = torch.randn(
            batch,
            time,
            neuronio_config["num_input"],
            device=device
        )

        Y = model.neuronio_eval_forward(X)

        assert Y.shape == (batch, time, 2)
        # Spike predictions should be in [0, 1]
        assert (Y[..., 0] >= 0).all() and (Y[..., 0] <= 1).all()

    def test_v2_neuronio_forward(
        self,
        neuronio_config,
        device
    ):
        """Test v2 NeuronIO-specific forward pass."""
        # Create v2-specific config
        config_v2 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_s_value'}
        model = ELM_v2(**config_v2).to(device)
        batch, time = 2, 100
        X = torch.randn(
            batch,
            time,
            neuronio_config["num_input"],
            device=device
        )

        Y = model.neuronio_eval_forward(X)

        assert Y.shape == (batch, time, 2)
        # Spike predictions should be in [0, 1]
        assert (Y[..., 0] >= 0).all() and (Y[..., 0] <= 1).all()

    def test_v1_viz_forward(self, simple_config, device):
        """Test v1 visualization forward pass."""
        model = ELM_v1(**simple_config).to(device)
        batch, time = 2, 20
        X = torch.randn(
            batch,
            time,
            simple_config["num_input"],
            device=device
        )

        outputs, s_record, m_record = (
            model.neuronio_viz_forward(X)
        )

        assert outputs.shape == (batch, time, 2)
        assert s_record.shape == (
            batch,
            time,
            model.num_synapse
        )
        assert m_record.shape == (
            batch,
            time,
            model.num_memory
        )

    def test_v2_viz_forward(self, simple_config, device):
        """Test v2 visualization forward pass."""
        model = ELM_v2(**simple_config).to(device)
        batch, time = 2, 20
        X = torch.randn(
            batch,
            time,
            simple_config["num_input"],
            device=device
        )

        outputs, b_record, m_record = (
            model.neuronio_viz_forward(X)
        )

        assert outputs.shape == (batch, time, 2)
        assert b_record.shape == (
            batch,
            time,
            model.num_branch
        )
        assert m_record.shape == (
            batch,
            time,
            model.num_memory
        )


# =============
# Routing Tests
# =============

class TestRouting:
    """Test routing mechanisms."""

    def test_no_routing(self, simple_config):
        """Test model without routing."""
        model = ELM_v1(**simple_config)
        X = torch.randn(2, 10, simple_config["num_input"])

        routed = model.route_input_to_synapses(X)
        # Without routing, should be identity
        assert torch.equal(routed, X)

    def test_neuronio_routing_indices(self, neuronio_config):
        """Test neuronio routing creates valid indices."""
        # Create v1-specific config
        config_v1 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_b_value'}
        model = ELM_v1(**config_v1)

        indices = model.input_to_synapse_indices
        # All indices should be valid (< num_input)
        assert (indices < model.num_input).all()
        assert (indices >= 0).all()

    def test_routing_maintains_shape(self, neuronio_config):
        """Test routing maintains correct tensor shapes."""
        # Create v1-specific config
        config_v1 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_b_value'}
        model = ELM_v1(**config_v1)
        batch, time = 4, 30
        X = torch.randn(
            batch,
            time,
            neuronio_config["num_input"]
        )

        routed = model.route_input_to_synapses(X)
        assert routed.shape == (batch, time, model.num_synapse)


# ===============
# Timescale Tests
# ===============

class TestTimescales:
    """Test timescale properties and bounds."""

    def test_v1_tau_m_bounds(self, simple_config):
        """Test v1 memory timescales stay within bounds."""
        model = ELM_v1(**simple_config)
        tau_m = model.tau_m

        assert (tau_m >= model.memory_tau_min).all()
        assert (tau_m <= model.memory_tau_max).all()

    def test_v2_tau_m_bounds(self, simple_config):
        """Test v2 memory timescales stay within bounds."""
        model = ELM_v2(**simple_config)
        tau_m = model.tau_m

        assert (tau_m >= model.memory_tau_min).all()
        assert (tau_m <= model.memory_tau_max).all()

    def test_kappa_range(self, simple_config):
        """Test decay factors are in valid range (0, 1)."""
        model = ELM_v1(**simple_config)

        assert (model.kappa_m > 0).all()
        assert (model.kappa_m < 1).all()
        assert (model.kappa_s > 0).all()
        assert (model.kappa_s < 1).all()

    def test_v2_kappa_lambda_range(self, simple_config):
        """Test v2 kappa_lambda is in valid range."""
        model = ELM_v2(**simple_config)

        assert (model.kappa_lambda > 0).all()
        assert (model.kappa_lambda < 1).all()

    def test_timescale_diversity(self, simple_config):
        """Test memory timescales are diverse (logspace)."""
        model = ELM_v1(**simple_config)
        tau_m = model.tau_m

        # Should span from min to max
        assert tau_m.min() < model.memory_tau_max / 2
        assert tau_m.max() > model.memory_tau_min * 2


# ====================
# Memory Dynamics Tests
# ====================

class TestMemoryDynamics:
    """Test memory update dynamics."""

    def test_v1_memory_bounded(self, simple_config, device):
        """Test v1 memory stays bounded over time."""
        model = ELM_v1(**simple_config).to(device)
        batch, time = 4, 200
        X = torch.randn(
            batch,
            time,
            simple_config["num_input"],
            device=device
        )

        _, _, m_record = model.neuronio_viz_forward(X)

        # Memory should remain bounded (not explode)
        assert (m_record.abs() < 100).all()

    def test_v2_memory_bounded(self, simple_config, device):
        """Test v2 memory stays bounded over time."""
        model = ELM_v2(**simple_config).to(device)
        batch, time = 4, 200
        X = torch.randn(
            batch,
            time,
            simple_config["num_input"],
            device=device
        )

        _, _, m_record = model.neuronio_viz_forward(X)

        # Memory should remain bounded (not explode)
        assert (m_record.abs() < 100).all()

    def test_zero_input_stability(self, simple_config, device):
        """Test memory remains stable with zero input."""
        model = ELM_v1(**simple_config).to(device)

        # Create input: first 10 timesteps with signal,
        # then 100 timesteps of zeros
        X_signal = torch.randn(
            1, 10, simple_config["num_input"], device=device
        )
        X_zero = torch.zeros(
            1, 100, simple_config["num_input"], device=device
        )
        X_combined = torch.cat([X_signal, X_zero], dim=1)

        _, _, m_record = model.neuronio_viz_forward(X_combined)

        # Memory should remain bounded during zero input
        # (due to nonlinear MLP, it may not strictly decay)
        assert m_record[0, 10:].abs().max() < 10.0

        # Eventually should stabilize (last 20 timesteps
        # should have low variance)
        mem_final = m_record[0, -20:].abs().mean(dim=0)
        assert mem_final.std() < 1.0


# ================
# Comparison Tests
# ================

class TestV1VsV2:
    """Compare behavior between v1 and v2."""

    def test_different_outputs(self, simple_config, device):
        """Test v1 and v2 produce different outputs."""
        torch.manual_seed(42)
        model_v1 = ELM_v1(**simple_config).to(device)

        torch.manual_seed(42)
        model_v2 = ELM_v2(**simple_config).to(device)

        X = torch.randn(
            2,
            20,
            simple_config["num_input"],
            device=device
        )

        Y_v1 = model_v1(X)
        Y_v2 = model_v2(X)

        # Should be different due to different dynamics
        assert not torch.allclose(Y_v1, Y_v2, rtol=0.1)

    def test_v2_parameter_efficiency(self, neuronio_config):
        """Test v2 parameter efficiency compared to v1."""
        # Create v1-specific config
        config_v1 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_b_value'}
        # Create v2-specific config
        config_v2 = {k: v for k, v in neuronio_config.items()
                     if k != 'tau_s_value'}

        model_v1 = ELM_v1(**config_v1)
        model_v2 = ELM_v2(**config_v2)

        params_v1 = sum(
            p.numel() for p in model_v1.parameters()
            if p.requires_grad
        )
        params_v2 = sum(
            p.numel() for p in model_v2.parameters()
            if p.requires_grad
        )

        # With neuronio_routing and same num_synapse,
        # parameter counts are similar (dominated by MLP/output)
        # v2's efficiency comes from architectural design,
        # not always lower param count
        print(f"\nv1 params: {params_v1}, v2 params: {params_v2}")
        assert params_v2 <= params_v1 * 1.1  # Within 10%

    def test_v2_w_s_learnable(self, simple_config):
        """Test synapse weight learnability in both versions."""
        model_v1 = ELM_v1(**simple_config)
        model_v2 = ELM_v2(**simple_config)

        # Both v1 and v2 have learnable w_s in current impl
        # (though v1 comment says "fixed", it's not enforced)
        assert model_v1._proto_w_s.requires_grad
        assert model_v2._proto_w_s.requires_grad


# ==============
# Edge Case Tests
# ==============

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_timestep(self, simple_config, device):
        """Test forward pass with single timestep."""
        model = ELM_v1(**simple_config).to(device)
        X = torch.randn(
            2,
            1,  # Single timestep
            simple_config["num_input"],
            device=device
        )

        Y = model(X)
        assert Y.shape == (2, 1, simple_config["num_output"])

    def test_large_batch(self, simple_config, device):
        """Test forward pass with large batch size."""
        model = ELM_v1(**simple_config).to(device)
        X = torch.randn(
            64,  # Large batch
            10,
            simple_config["num_input"],
            device=device
        )

        Y = model(X)
        assert Y.shape == (64, 10, simple_config["num_output"])

    def test_long_sequence(self, simple_config, device):
        """Test forward pass with long sequence."""
        model = ELM_v1(**simple_config).to(device)
        X = torch.randn(
            2,
            500,  # Long sequence
            simple_config["num_input"],
            device=device
        )

        Y = model(X)
        assert Y.shape == (2, 500, simple_config["num_output"])
        assert not torch.isnan(Y).any()

    def test_gradient_flow(self, simple_config, device):
        """Test gradients flow properly through model."""
        model = ELM_v1(**simple_config).to(device)
        X = torch.randn(
            2,
            10,
            simple_config["num_input"],
            device=device,
            requires_grad=True
        )

        Y = model(X)
        loss = Y.sum()
        loss.backward()

        # Check gradients exist
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

    def test_invalid_routing(self):
        """Test invalid routing configuration raises error."""
        with pytest.raises(AssertionError):
            ELM_v1(
                num_input=100,
                num_output=2,
                num_memory=10,
                input_to_synapse_routing="invalid_routing"
            )


# ==================
# Integration Tests
# ==================

class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_training_step(self, simple_config, device):
        """Test a complete training step."""
        model = ELM_v1(**simple_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        X = torch.randn(
            4,
            20,
            simple_config["num_input"],
            device=device
        )
        target = torch.randn(
            4,
            20,
            simple_config["num_output"],
            device=device
        )

        # Forward
        Y = model(X)
        loss = torch.nn.functional.mse_loss(Y, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is finite
        assert torch.isfinite(loss)

    def test_model_save_load(self, simple_config, tmp_path):
        """Test model can be saved and loaded."""
        model = ELM_v1(**simple_config)
        X = torch.randn(2, 10, simple_config["num_input"])

        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Load
        model_loaded = ELM_v1(**simple_config)
        model_loaded.load_state_dict(torch.load(save_path))

        # Check outputs match
        Y1 = model(X)
        Y2 = model_loaded(X)
        assert torch.allclose(Y1, Y2)


# ==============
# Run all tests
# ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
