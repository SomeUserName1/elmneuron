# Methods for Computing Lyapunov Exponents and Assessing Chaotic Dynamics in ELM Neurons

## Overview

This document summarizes relevant methods from research papers on Lyapunov exponents and chaos analysis in dynamical systems, particularly applicable to investigating whether the ELM neuron exhibits chaotic dynamics near spike initiation.

## 1. Lyapunov Exponents: Theory and Interpretation

### 1.1 Definition and Meaning

The Lyapunov exponent (LE) measures the average sensitivity of a dynamical system to perturbations in initial conditions:

- **Positive LEs**: Indicate exponential divergence of trajectories → chaotic dynamics
- **Negative LEs**: Indicate exponential convergence of trajectories → stable dynamics
- **Zero LE**: Marginal stability

The full **Lyapunov spectrum** λ₁ ≥ λ₂ ≥ ... ≥ λₙ characterizes the system's behavior with respect to infinitesimal perturbations across all dimensions.

### 1.2 Edge of Chaos Hypothesis

Systems may operate optimally as information processing devices near a transition between order and chaos—the "edge of chaos." This regime supports:
- Rich repertoire of distinct state sequences
- Intermediate-term memory of initial conditions
- Both fading-memory property AND separation property

## 2. Computational Methods for Lyapunov Exponents

### 2.1 Jacobian-Based Method

**Core Approach** (from Benettin et al., 1980):

1. **Compute the Jacobian matrix** of the dynamical system:
   ```
   J(t) = ∂f/∂x evaluated along a trajectory
   ```

2. **For continuous-time systems** with state evolution dx/dt = f(x):
   ```
   J_ij(t) = ∂f_i/∂x_j
   ```

3. **For discrete-time/event-driven systems** (e.g., spiking neurons):
   - Use single-spike Jacobians D(t_s) that describe evolution of infinitesimal phase perturbations
   - D(t_s) = d𝜑̃(t_s) / d𝜑̃(t_{s-1})

4. **For RNNs with tanh activation**:
   ```
   J_ij(t) = -δ_ij/τ + g/√N W_ij φ'(h_i(t))
   ```
   where φ' = sech²(h_i) for tanh activation

### 2.2 Reorthonormalization Procedure (QR Decomposition)

This is the standard numerical algorithm for computing the full Lyapunov spectrum:

**Algorithm**:

1. Initialize N orthonormal perturbation vectors v₁, v₂, ..., vₙ

2. **For each time step** (or spike event):

   a. **Propagate** perturbations using the Jacobian:
      ```
      v_i(t+Δt) = J(t) · v_i(t)
      ```

   b. **Reorthonormalize** using QR decomposition:
      - Form matrix V from column vectors [v₁, v₂, ..., vₙ]
      - Compute V = QR (QR decomposition)
      - Set v_i ← columns of Q
      - Accumulate diagonal elements of R

   c. **After T time steps**, Lyapunov exponents are:
      ```
      λ_i = (1/T) Σ log(r_ii)
      ```
      where r_ii are diagonal elements from QR steps

3. **Key parameters**:
   - Initial perturbation size: ~10⁻¹² (small enough to avoid numerical precision issues)
   - Reorthonormalization frequency: t_ONS (every few time steps)
   - Total simulation time: t_sim (long enough for convergence)

### 2.3 Mean Lyapunov Exponent

The mean Lyapunov exponent represents the rate of phase-space volume compression:

```
λ_mean = (1/N) Σᵢ λ_i = lim_{t→∞} (1/t) log|det J(t)|
```

For rate-based RNNs, random matrix theory predicts:
```
λ_mean = -1/τ + (g²/2) ⟨φ'²⟩
```

## 3. Perturbation Analysis Methods

### 3.1 Small Perturbations to State Variables

**Method**: Perturb membrane potentials/activations by small amount δ

1. Run reference trajectory with initial condition x₀
2. Run perturbed trajectory with initial condition x₀ + δv (|δ| ≪ 1)
3. Measure distance evolution: d(t) = |x_pert(t) - x_ref(t)|
4. Compute local Lyapunov exponent: λ_local = (1/t) log(d(t)/d(0))

**For ELM neurons**: Perturb memory state m_t or branch/synapse states b_t/s_t

### 3.2 Single-Spike Perturbations

**Method** (particularly relevant for spiking systems):

1. Inject a single extra spike at time t₀ in one input channel
2. Track how this perturbation affects:
   - Output spike timing
   - Internal state variables
   - Population activity
3. Measure decorrelation over time

**Key insight**: Even with negative Lyapunov spectrum (stable dynamics), single-spike perturbations can cause complete microstate decorrelation in spiking networks due to "flux tube" structure of phase space.

### 3.3 Input Perturbation Analysis

**Method for fitted models**:

1. Take trained ELM neuron with learned parameters
2. Generate reference trajectory with input sequence u(t)
3. Add small perturbation: ũ(t) = u(t) + ε·η(t), where:
   - ε is perturbation magnitude (vary systematically)
   - η(t) is perturbation pattern (e.g., Gaussian noise)
4. Measure output divergence over time windows
5. Repeat for different ε values and time points (especially near spike initiation)

## 4. Application to ELM Neurons

### 4.1 Recommended Approach

**For ELM v1**:
1. Compute Jacobian with respect to state variables [s_t, m_t]:
   ```
   J = [∂s_{t+1}/∂s_t    ∂s_{t+1}/∂m_t  ]
       [∂m_{t+1}/∂s_t    ∂m_{t+1}/∂m_t  ]
   ```

2. Include MLP Jacobian for memory update:
   ```
   ∂Δm_t/∂[b_inp, κ_m·m_{t-1}]
   ```

**For ELM v2** (Branch-ELM):
1. Compute Jacobian with respect to [b_t, m_t]:
   ```
   J = [∂b_{t+1}/∂b_t    ∂b_{t+1}/∂m_t  ]
       [∂m_{t+1}/∂b_t    ∂m_{t+1}/∂m_t  ]
   ```

2. Account for learnable synaptic weights w_s in perturbation analysis

### 4.2 Specific Test Cases

**Near spike initiation**:
1. Identify time points just before predicted spikes
2. Apply perturbations of varying magnitudes
3. Measure output sensitivity
4. Compute local Lyapunov exponents
5. Compare behavior near-spike vs. away-from-spike

**Throughout trajectory**:
1. Apply uniform perturbations across entire time series
2. Generate Lyapunov spectrum
3. Check for positive exponents (chaos indicators)

### 4.3 PyTorch Implementation Considerations

- Use `torch.autograd` to compute Jacobians automatically
- Implement QR decomposition using `torch.linalg.qr()`
- Handle JIT-compiled methods carefully (may need native Python version)
- Batch perturbations for computational efficiency

## 5. Related Concepts

### 5.1 Kolmogorov-Sinai Entropy Rate (H)

**Definition**: Quantifies the dynamical entropy rate - the information production rate due to sensitivity to initial conditions.

**Formula**:
```
H = Σ λ_i for all λ_i > 0
```
Sum of all positive Lyapunov exponents.

**Physical Interpretation**:
- **Measures**: Amplification of uncertainty / speed at which microscopic perturbations affect macroscopic fluctuations
- **Constraint on information processing**: Given finite precision, sensitive dependence makes long-term predictions impossible
- **Neural coding perspective**: Can contribute to "noise entropy" because dynamic amplification of microscopic noise impairs coding capacity
- **Diversity of states**: Characterizes complexity of state sequences the system can generate

**Key Properties**:
- **Invariant under diffeomorphisms** of phase space
- **Only known general method** for calculating entropy of high-dimensional differentiable dynamical systems
- **Extensive**: Grows linearly with system size N in chaotic networks
- **Sampling-based methods fail**: Grassberger-Procaccia algorithm requires data that scales exponentially with dimensionality

**Non-Monotonic Behavior with Coupling Strength**:

For recurrent neural networks, entropy rate shows surprising non-monotonic behavior:

```
H increases → peaks at intermediate g → decreases at large g
```

Where g is coupling strength. This reflects:
- Near onset of chaos: More unstable directions → higher entropy
- Far from onset: Saturation of dynamics → fewer effective unstable directions → lower entropy
- **Key insight**: Largest Lyapunov exponent λ_max keeps growing with g, but entropy rate H decreases!

**Effects of External Factors**:
- **Time discretization**: Increases entropy rate
- **Time-varying input**: Drastically **reduces** entropy rate
- **Noise**: Dynamic amplification can impair coding capacity

**Computation**:
```python
def kolmogorov_sinai_entropy(lyapunov_spectrum):
    """
    Compute entropy rate from Lyapunov spectrum

    Args:
        lyapunov_spectrum: array of Lyapunov exponents (sorted descending)
    Returns:
        H: entropy rate (bits/time or nats/time depending on log base)
    """
    positive_exponents = lyapunov_spectrum[lyapunov_spectrum > 0]
    H = np.sum(positive_exponents)
    return H
```

**For ELM Neurons**:
- Compute from local Lyapunov spectrum near spike times
- Compare H near spikes vs. away from spikes
- High H → unpredictable spike timing (chaotic)
- Low H → more predictable dynamics (stable or "stable chaos")
- Non-monotonic dependence on lambda_value parameter?

### 5.2 Attractor Dimension (D)

**Definition**: Effective dimensionality of the attractor - the low-dimensional manifold toward which the system evolves.

**Kaplan-Yorke Dimension Formula**:
```
D_KY = k + (Σᵢ₌₁ᵏ λ_i) / |λ_{k+1}|
```
where k is the largest integer such that Σᵢ₌₁ᵏ λ_i ≥ 0

**Physical Interpretation**:
- **Measures**: Diversity of collective network activity states
- **Characterizes**: Lower-dimensional manifold in high-dimensional phase space
- **Much smaller than phase space**: D ≪ N (number of phase space dimensions)
- **Challenge**: Identifying this lower-dimensional structure is difficult without Lyapunov analysis

**Key Properties**:
- **Invariant under diffeomorphisms** of phase space
- **Extensive**: Scales linearly with system size N in chaotic networks
- **Obtained from Lyapunov spectrum**: No need for sampling-based methods
- **Correlation dimension** D₂ ≈ D_KY (approximately equal to Kaplan-Yorke for many systems)

**Non-Monotonic Behavior** (same as entropy):

```
D increases near onset of chaos → peaks → decreases far from onset
```

This parallels entropy rate behavior:
- Intermediate g: Maximum diversity of states
- Large g: Saturation reduces effective dimensionality
- Reflects number of unstable directions

**Comparison with Phase Space Dimensions**:
- Phase space: N dimensions (total state variables)
- Attractor: D ≪ N dimensions (effective dynamics)
- Example: For N=10,000 rate network, D might be ~1,000-3,000
- Indicates **low-dimensional attractor** despite high-dimensional phase space

**Why Sampling Methods Fail**:
- Grassberger-Procaccia algorithm estimates correlation dimension D₂
- **Intractable for high dimensions**: Data required scales as exp(D)
- For D=1000, would need impossibly large dataset
- **Solution**: Compute from Lyapunov spectrum (the only tractable method)

**Effects of External Factors**:
- **Time-varying input**: Drastically reduces dimensionality D
- **Discretization**: Can increase effective dimensionality
- **Network size**: D grows with N but D/N remains approximately constant (extensive)

**Computation**:
```python
def kaplan_yorke_dimension(lyapunov_spectrum):
    """
    Compute Kaplan-Yorke dimension from Lyapunov spectrum

    Args:
        lyapunov_spectrum: array sorted in descending order
    Returns:
        D_KY: Kaplan-Yorke dimension
    """
    cumsum = np.cumsum(lyapunov_spectrum)
    # Find largest k where cumsum is non-negative
    k = np.where(cumsum >= 0)[0]

    if len(k) == 0:
        return 0  # All exponents negative

    k = k[-1]  # Largest index where cumsum >= 0

    if k == len(lyapunov_spectrum) - 1:
        return len(lyapunov_spectrum)  # All exponents contribute

    # Kaplan-Yorke formula
    D_KY = k + 1 + cumsum[k] / abs(lyapunov_spectrum[k + 1])
    return D_KY
```

**For ELM Neurons**:
- Compute effective dimensionality of memory dynamics
- Compare to number of memory units d_m
- D < d_m suggests dynamics constrained to lower-dimensional manifold
- D ≈ d_m suggests full utilization of memory capacity
- Local changes in D near spike initiation?
- Relationship to memory capacity: MC vs. D vs. d_m

**Connection to Information Processing**:
- **Ergodic theory**: D relates to diversity of accessible states
- **Computational capacity**: Higher D (up to a point) → richer dynamics
- **Trade-off**: Very high D may indicate excessive sensitivity (poor generalization)
- **Optimal regime**: Intermediate D balances richness and stability

### 5.3 Stable Chaos

Phenomenon where:
- All Lyapunov exponents are negative (stable dynamics)
- But single perturbations can cause complete decorrelation
- Due to "flux tube" structure in phase space
- Radius of stability tubes may decrease with system size

## 6. Key References

1. **Benettin et al. (1980)**: "Lyapunov Characteristic Exponents for Smooth Dynamical Systems" - fundamental algorithm

2. **Monteforte & Wolf (2012)**: "Dynamic Flux Tubes, Tanh Networks, Stable Chaos" - stable chaos in neural networks

3. **Boedecker et al. (2010)**: "Information Processing in Echo State Networks" - criticality and Lyapunov exponents

4. **Phys. Rev. Research (2023)**: Modern methods for RNNs with extensive chaos

## 7. Practical Workflow for ELM Investigation

### Step 1: Load Fitted Model
```python
model = ELM_v2.load_state_dict(torch.load("models/elm_dm15.pt"))
model.eval()
```

### Step 2: Generate Reference Trajectory
```python
with torch.no_grad():
    output_ref = model(input_sequence)
```

### Step 3: Implement Jacobian Computation
```python
def compute_jacobian(model, state, input_t):
    # Enable gradient computation
    # Compute J = ∂state_{t+1}/∂state_t
    pass
```

### Step 4: Implement Lyapunov Calculation
```python
def lyapunov_spectrum(model, input_seq, n_exponents, t_ons=10):
    # QR decomposition method
    pass
```

### Step 5: Analyze Results
- Plot Lyapunov spectrum
- Identify chaotic regions (positive exponents)
- Correlate with spike prediction errors
- Generate visualizations

## 8. Information-Theoretic Measures

### 8.1 Active Information Storage (AIS)

**Definition**: Measures the stored information that is currently being used to compute the next state of a node.

For a node X with history x^(k)_n = {x_n, x_{n-1}, ..., x_{n-k+1}}:

```
A_X(k) = I(x^(k)_n; x_{n+1}) = lim_{k→∞} I(x^(k)_n; x_{n+1})
```

**Physical Interpretation**:
- Quantifies how much information from the past is relevant for predicting the future
- High AIS: Strong memory/storage capability
- Low AIS: Little memory retention
- Can be facilitated in a distributed fashion via neighbors (stigmergy)

**For ELM Neurons**:
- Measure AIS for memory units m_t
- Measure AIS for branch/synapse states b_t or s_t
- Compare AIS at different timescales
- Check if AIS peaks near edge of chaos

### 8.2 Transfer Entropy (TE)

**Definition**: Measures information transfer from source Y to destination X, accounting for X's own history.

```
T_{Y→X}(k) = I(y_n; x_{n+1} | x^(k)_n)
           = H(x_{n+1} | x^(k)_n) - H(x_{n+1} | x^(k)_n, y_n)
```

**Physical Interpretation**:
- Information provided by source about destination's next state
- NOT contained in destination's past
- Addresses directionality (unlike mutual information)
- Measures causal information flow

**For ELM Neurons**:
- TE from inputs to memory: T_{input→memory}
- TE between memory units: T_{m_i→m_j}
- TE from branches to memory (v2 only): T_{branch→memory}
- Network of information flows

**Implementation Notes**:
- Use conditional mutual information
- Kernel estimation for probability distributions
- Fixed radius kernel: r ≈ 0.2 (typical value)
- History length k: typically k=2 sufficient for discrete-time systems

### 8.3 Memory Capacity (MC)

**Definition**: Quantifies the short-term memory capability of a dynamical system.

For delay-k task (predicting input delayed by k steps):

```
MC_k = cov²(u(t-k), y_k(t)) / (var(u(t-k)) · var(y_k(t)))
```

Total memory capacity:
```
MC_total = Σ_k MC_k
```

**Theoretical Upper Bound**:
- For linear systems: MC ≤ N (number of nodes)
- For nonlinear systems with fading memory: MC < N

**For ELM Neurons**:
1. Train separate readouts for each delay k
2. Compute squared correlation between predicted and actual delayed inputs
3. Sum over all delays with significant correlation
4. Compare memory capacity to number of memory units (d_m)

**Practical Protocol**:
- Input: uniformly random time series u(t) ∈ [-1, 1]
- Outputs: y_k(t) trained to predict u(t-k) for k=1,2,...,K_max
- Use linear regression for readout training
- Test on separate data

### 8.4 Computational Properties at Criticality

**Fading Memory Property**:
- Information about differences in microstate decays over time
- Essential for temporal integration
- Too strong: overly sensitive to initial conditions
- Too weak: cannot distinguish different inputs

**Separation Property**:
- Distinguishable inputs lead to significantly different internal states
- Necessary for stimulus discrimination
- Realized through exponentially separating trajectories

**Coexistence**: Papers show that fading memory AND separation can coexist through "flux tube" structure:
- Within flux tube: fading memory (convergence to stable trajectory)
- Between flux tubes: separation property (exponential divergence)
- Flux tube radius ε_FT scales as: ε_FT ∝ 1/(√(KN) · firing_rate · τ_m)

### 8.5 Kernel Quality vs. Generalization

**Kernel Quality**: Measures how well the reservoir separates different inputs
- High kernel quality → good separation of states
- Related to richness of reservoir dynamics

**Generalization Ability**: Measures how well learned mappings generalize
- High generalization → robust readout across test data
- Related to stability of reservoir dynamics

**Trade-off at Edge of Chaos**:
```
Computation ∝ f(Kernel_Quality, Generalization)
```
- Ordered regime: Good generalization, poor kernel quality
- Chaotic regime: Good kernel quality, poor generalization
- Critical regime: Optimal balance

### 8.6 Practical Computation

**For Active Information Storage**:
```python
def compute_AIS(state_history, k=2):
    """
    Args:
        state_history: array of shape (T, d_state)
        k: history length
    Returns:
        AIS: array of shape (d_state,)
    """
    # Estimate I(x^(k)_n; x_{n+1}) using kernel estimation
    # Or use discrete binning for continuous variables
    pass
```

**For Transfer Entropy**:
```python
def compute_TE(source, destination, k=2):
    """
    Args:
        source: array of shape (T,)
        destination: array of shape (T,)
        k: history length
    Returns:
        TE: scalar value
    """
    # Compute I(y_n; x_{n+1} | x^(k)_n)
    # Use conditional mutual information
    pass
```

**For Memory Capacity**:
```python
def compute_MC(reservoir_states, input_signal, max_delay=100):
    """
    Args:
        reservoir_states: array of shape (T, d_m)
        input_signal: array of shape (T,)
        max_delay: maximum delay to test
    Returns:
        MC_k: array of shape (max_delay,)
        MC_total: scalar
    """
    MC_k = []
    for k in range(1, max_delay+1):
        # Train linear readout to predict input_signal[t-k]
        # Compute squared correlation
        # Append to MC_k
        pass
    return np.array(MC_k), np.sum(MC_k)
```

### 8.7 Entropy of State Sequences

From flux tube analysis, the entropy of distinct spike/state sequences scales as:

```
H ≈ N · log(√(KN) · ⟨ν⟩ · τ_m)
```

Where:
- N: number of neurons/units
- K: connectivity
- ⟨ν⟩: average firing rate / activation frequency
- τ_m: memory timescale

**Implications**:
- Entropy grows faster than extensive (> N)
- Depends on both connectivity and dynamics
- Time to display all sequences: T ~ (√(KN)·⟨ν⟩·τ_m)^(N-1) / (N·⟨ν⟩)

## 9. Application to Chaos Analysis in ELM

### 9.1 Integrated Analysis Pipeline

Combine Lyapunov analysis with information-theoretic measures:

1. **Compute Lyapunov spectrum** → characterize stability
2. **Measure AIS** → quantify memory/storage
3. **Measure TE** → map information flows
4. **Compute MC** → assess temporal integration
5. **Compare near vs. away from spikes** → localize chaos

### 9.2 Hypothesis Testing

**If ELM exhibits chaotic dynamics near spike initiation**:
- Lyapunov spectrum: Local positive exponents near spike times
- AIS: Decreased near spikes (less predictable from past)
- TE: Increased from recent inputs (more sensitive to external drive)
- MC: Reduced temporal integration during spike generation

**If ELM exhibits "stable chaos"**:
- Lyapunov spectrum: All negative globally
- But: High sensitivity to single-spike perturbations
- AIS: High overall (good memory)
- TE: Structured information flow
- MC: Good memory capacity
- State separation: Rapid decorrelation after discrete events

### 9.3 Expected Signatures

**Near spike initiation** (if chaotic):
```
λ_max > 0 (locally)
AIS ↓
TE_{input→memory} ↑
Sensitivity to perturbations ↑
```

**Away from spikes** (if stable):
```
λ_max < 0
AIS moderate/high
TE structured
MC ~ d_m (number of memory units)
```

### 9.4 Comparison with Reservoir Computing

ELM neurons can be viewed as reservoir computing systems where:
- **Reservoir**: Synaptic/branch layer + memory dynamics
- **Readout**: Output weights W_y
- **Tuning parameter**: Could vary λ (lambda_value) or timescales

Expected behavior:
- Well-tuned ELM: Operates near edge of chaos for good computational properties
- Over-tuned: Too chaotic, poor generalization, spike errors
- Under-tuned: Too stable, poor separation, limited expressiveness

## Conclusion

The combination of Jacobian-based methods, perturbation analysis, and information-theoretic measures provides a comprehensive toolkit for assessing whether ELM neurons exhibit chaotic dynamics. The key innovation will be adapting these methods to the specific structure of ELM v1/v2, particularly handling the hierarchical computation and learnable parameters.

**Three complementary perspectives**:
1. **Dynamical Systems**: Lyapunov exponents, phase space structure, attractors
2. **Information Theory**: AIS, TE, MC, entropy of sequences
3. **Computational**: Separation vs. fading memory, kernel quality vs. generalization

Based on the papers reviewed, special attention should be paid to:
1. Local dynamics near spike times (where prediction errors occur)
2. The possibility of "stable chaos" (negative spectrum but sensitivity to discrete events)
3. Scaling of sensitivity with model dimensions (d_m, number of branches/synapses)
4. Comparison between different timescales (τ_m, τ_b, τ_s)
5. Information storage and transfer patterns during spike generation
6. Memory capacity relative to model size
7. Trade-off between separation and fading memory properties
