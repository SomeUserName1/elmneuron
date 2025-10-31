# Chaos Analysis of ELM Neurons - Validation Guide

This guide helps you validate the chaos analysis code method-by-method.

## Overview

The script `analyze_chaos_elm.py` implements all methods from `methods_summary.md` to investigate whether ELM neurons exhibit chaotic dynamics near spike initiation.

## Quick Start

```bash
python analyze_chaos_elm.py \
    --model_path models/elm_dm15.pt \
    --dataset_path /path/to/neuronio/data \
    --output_dir chaos_analysis_results \
    --sample_idx 0
```

## Method-by-Method Validation

### SECTION 1: Data Loading and Model Setup

**What it does:**
- Loads NeuronIO dataset sample
- Loads pre-trained ELM v2 model
- Generates predictions
- Visualizes input, targets, and predictions

**To validate:**
1. Check that input dimensions match (1278 inputs for NeuronIO)
2. Verify target shape is (T, 2) for [spikes, soma_voltage]
3. Examine `step1_data_sample.png` for data quality

**Expected output:**
```
✓ Saved: step1_data_sample.png
```

**Potential issues:**
- **Data format mismatch**: Adjust `NeuronIOLoader.load_sample()` to match actual HDF5 structure
- **Model architecture**: Update model initialization parameters if different

---

### SECTION 2: Lyapunov Spectrum Computation

**What it does:**
- Computes Jacobian matrix at each time step
- Uses QR decomposition for reorthonormalization
- Accumulates Lyapunov exponents
- Sorts spectrum in descending order

**Implementation status:** ⚠️ **REQUIRES COMPLETION**

The Jacobian computation is currently a placeholder. You need to:

1. **Extract internal states** from ELM model:
```python
def get_internal_state(model, inputs):
    # Modify model to return [b_t, m_t]
    # b_t: branch activations (num_branch,)
    # m_t: memory states (num_memory,)
    pass
```

2. **Implement state transition function**:
```python
def state_transition(state, input_t):
    # Given current state and input, compute next state
    # Using ELM v2 equations:
    # b_{t+1} = κ_b * b_t + Σ(w_s * x_t per branch)
    # m_{t+1} = κ_m * m_t + (1 - κ_λ) * MLP([b_t, κ_m*m_{t-1}])
    pass
```

3. **Compute Jacobian using autograd**:
```python
J = torch.autograd.functional.jacobian(
    lambda s: state_transition(s, input_t),
    state
)
```

**To validate:**
- Check spectrum shape: should be (n_exponents,)
- Verify sorting: spectrum[0] >= spectrum[1] >= ...
- Sanity check: mean exponent should be negative for stable system
- Compare to analytical predictions if available

**Expected output:**
```
✓ Saved: step2_lyapunov_spectrum.png
  λ_max = -0.0234
  # positive exponents = 0
```

**Diagnostic checks:**
- If all exponents are large and negative: τ might be too small
- If exponents don't converge: increase t_sim or t_ons
- If results vary wildly: check numerical stability (use float64)

---

### SECTION 3: Entropy and Dimensionality

**What it does:**
- Computes Kolmogorov-Sinai entropy: H = Σ λᵢ⁺
- Computes Kaplan-Yorke dimension: D = k + (Σ λᵢ) / |λₖ₊₁|
- Visualizes both metrics

**To validate:**
1. Verify H = sum of positive exponents
2. Check that D is computed correctly:
   - Find largest k where cumsum(λ) >= 0
   - Apply formula
3. Ensure D < phase space dimension

**Expected relationships:**
- H = 0 if all λᵢ ≤ 0 (stable dynamics)
- H > 0 indicates chaos
- D ≪ N indicates low-dimensional attractor

**Expected output:**
```
✓ Saved: step3_entropy_dimension.png
  Entropy rate H = 0.0000 nats/time
  Attractor dimension D = 8.45
  Phase space dimension = 60
```

**Diagnostic checks:**
- If D ≈ N: system using full phase space (unusual)
- If D = 0: all exponents negative (expected for stable)
- If H is large: check for numerical errors

---

### SECTION 4: Local Analysis (Near Spikes)

**What it does:**
- Computes local λ_max in sliding windows
- Identifies spike times
- Compares dynamics near spikes vs away from spikes

**To validate:**
1. Verify spike detection threshold (default 0.5)
2. Check window size is appropriate (default 50 steps)
3. Examine temporal correlation between λ_max and spikes

**Expected behavior:**
- If chaotic near spikes: λ_max increases before/during spikes
- If stable chaos: λ_max remains negative but divergence increases

**Expected output:**
```
✓ Saved: step4_local_chaos.png
  λ_max near spikes: -0.015 ± 0.005
  λ_max away from spikes: -0.025 ± 0.003
```

**Key questions:**
- Is λ_max more positive near spikes?
- Is variance higher near spikes?
- Are there systematic patterns?

---

### SECTION 5: Perturbation Analysis

**What it does:**
- Adds small Gaussian noise to inputs (ε = 1e-3)
- Measures output divergence over time
- Compares to spike times

**To validate:**
1. Check that ε is appropriately small
2. Verify divergence grows over time
3. Look for exponential growth (linear in log plot)

**Expected behavior:**
- Stable system: divergence saturates
- Chaotic system: exponential divergence
- Stable chaos: rapid initial divergence then saturation

**Expected output:**
```
✓ Saved: step5_perturbations.png
  Mean final divergence: 0.001234
  Max divergence: 0.005678
```

**Diagnostic checks:**
- If divergence is huge: ε might be too large
- If divergence is tiny: ε might be too small or system very stable
- Check log plot for exponential signature

---

### SECTION 6: Active Information Storage (AIS)

**What it does:**
- Computes I(x^(k)_n; x_{n+1}) for each state variable
- Measures information from past used to predict future
- Separate analysis for memory and branch states

**Implementation status:** ⚠️ **REQUIRES STATE EXTRACTION**

Currently uses random data. You need to:

1. **Extract state history** from model forward pass:
```python
def extract_states(model, inputs):
    # Return memory_history: (T, num_memory)
    # Return branch_history: (T, num_branch)
    pass
```

2. **Improve MI estimation** (optional):
   - Current uses histogram binning
   - Consider KDE or k-NN methods for better estimates

**To validate:**
1. Check AIS values are reasonable (0-10 bits typical)
2. Compare memory vs branch AIS
3. Verify units have different AIS (heterogeneity expected)

**Expected output:**
```
✓ Saved: step6_AIS.png
  Mean AIS (memory): 2.34 bits
  Mean AIS (branches): 1.89 bits
```

**Expected patterns:**
- Memory units: Higher AIS (more storage)
- Branches: Lower AIS (more transient)
- Near edge of chaos: AIS peaks

---

### SECTION 7: Memory Capacity

**What it does:**
- Tests ability to predict delayed inputs
- Trains linear readouts for each delay k
- Computes MC_k = correlation²(target, prediction)
- Sums to get total memory capacity

**Implementation status:** ⚠️ **REQUIRES STATE EXTRACTION**

Currently uses random reservoir states. You need to extract actual states.

**To validate:**
1. Check MC_k decreases with delay
2. Verify MC_total ≤ N (theoretical upper bound)
3. Compare MC_total to d_m (number of memory units)

**Expected output:**
```
✓ Saved: step7_memory_capacity.png
  Total memory capacity: 12.45
  Number of memory units: 15
  MC / d_m ratio: 0.83
```

**Expected patterns:**
- MC_total ~ d_m indicates good utilization
- MC_total ≪ d_m indicates underutilization
- MC_total > d_m indicates distributed memory

---

### SECTION 8: Summary Report

**What it does:**
- Aggregates all metrics
- Provides interpretation
- Saves text summary

**To validate:**
1. Check all metrics are present
2. Verify interpretations are correct
3. Compare to known behaviors

**Expected output:**
```
✓ Saved: summary_report.txt
```

---

## Complete Validation Checklist

- [ ] Step 1: Data loads correctly, shapes match
- [ ] Step 2: Lyapunov spectrum computed (requires Jacobian implementation)
- [ ] Step 3: Entropy and dimension calculated correctly
- [ ] Step 4: Local analysis shows patterns near spikes
- [ ] Step 5: Perturbation analysis shows divergence
- [ ] Step 6: AIS computed (requires state extraction)
- [ ] Step 7: Memory capacity measured (requires state extraction)
- [ ] Step 8: Summary report generated

## Common Issues and Solutions

### Issue: "Jacobian computation not implemented"
**Solution:** Complete the `compute_jacobian()` method by:
1. Extracting internal states [b_t, m_t]
2. Defining state transition function
3. Using torch.autograd.functional.jacobian

### Issue: "State extraction returns wrong shapes"
**Solution:** Check ELM model structure:
```python
print(model.num_branch)  # Should match branch_history.shape[1]
print(model.num_memory)  # Should match memory_history.shape[1]
```

### Issue: "Computation is too slow"
**Solution:**
- Reduce sequence length (use first 500 steps)
- Reduce n_exponents (compute top 10-20 only)
- Use GPU if available (`--device cuda`)
- Increase t_ons (reorthonormalize less frequently)

### Issue: "Numerical instability"
**Solution:**
- Use double precision: `model.double()`, `inputs.double()`
- Check for NaN/Inf: `torch.isnan(J).any()`
- Reduce perturbation sizes
- Increase warmup period

### Issue: "Results don't make sense"
**Solution:**
- Verify model is trained (check loss/metrics)
- Check data preprocessing matches training
- Validate against simple test cases
- Compare multiple samples

## Interpreting Results

### Evidence of Chaos Near Spikes:

**Strong evidence:**
- λ_max increases near spikes (becomes positive or less negative)
- H increases locally near spikes
- Perturbation divergence accelerates near spikes
- AIS decreases near spikes

**Weak evidence:**
- λ_max remains negative everywhere (stable chaos)
- High perturbation sensitivity despite negative exponents
- Rapid decorrelation of spike timing

### Evidence of Stable Dynamics:

- All λᵢ ≤ 0 (negative spectrum)
- H = 0 (zero entropy rate)
- Perturbations decay exponentially
- High AIS throughout
- MC ~ d_m (good memory capacity)

### Edge of Chaos Indicators:

- λ_max ≈ 0 (near critical transition)
- Intermediate H and D values
- Balance between AIS and perturbation sensitivity
- Non-monotonic behavior with parameters

## Next Steps After Validation

1. **Complete missing implementations** (Jacobian, state extraction)
2. **Run on multiple samples** to verify consistency
3. **Vary model parameters** (lambda_value, timescales) to see effects
4. **Compare ELM v1 vs v2** architectures
5. **Analyze different spike types** (isolated vs bursts)
6. **Correlate with prediction errors** from evaluation metrics

## References

See `methods_summary.md` for detailed theoretical background and formulas.

## Support

If you encounter issues:
1. Check error messages carefully
2. Verify data/model paths
3. Try with smaller data first (first 100 time steps)
4. Use verbose printing for debugging
5. Compare intermediate results with expected values
