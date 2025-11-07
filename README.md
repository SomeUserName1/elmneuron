# The ELM: an Efficient and Expressive Phenomenological Neuron Model Can Solve Long-Horizon Tasks

This repository features a minimal implementation of the (Branch) Expressive Leaky Memory neuron in PyTorch.
Notebooks to train and evaluate on NeuronIO are provided, as well as pre-trained models of various sizes.

![The ELM neuron](./elm_neuron_sketch.png)

## Installation

1. Create the conda environment with `conda env create -f elm_env.yml`
2. Once installed, activate the environment with `conda activate elm_env`

## Models

The __models__ folder contains various sized Branch-ELM neuron models pre-trained on NeuronIO.

|  $d_m$    | 1      | 2      | 3      | 5      | 7      | 10     | 15     | 20     | 25     | 30     | 40     | 50     | 75     | 100    |
|-----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| #params   | 4601   | 4708   | 4823   | 5077   | 5363   | 5852   | 6827   | 8002   | 9377   | 10952  | 14702  | 19252  | 34127  | 54002  |
| AUC       | 0.9437 | 0.9582 | 0.9558 | 0.9757 | 0.9827 | 0.9878 | 0.9915 | 0.9922 | 0.9926 | 0.9929 | 0.9934 | 0.9934 | 0.9938 | 0.9935 |

We also include a best effort trained ELM neuron that achieves 0.9946 AUC with $d_m=100$.

## Notebooks

- __train_elm_on_shd.ipynb__: train an ELM neuron on SHD or SHD-Adding Dataset.
- __train_elm_on_neuronio.ipynb__: train an ELM neuron on NeuronIO Dataset.
- __eval_elm_on_neuronio.ipynb__: evaluate provided models on the NeuronIO Dataset.
- __neuronio_train_script__: script to train an ELM neuron on NeuronIO Dataset.

## Code

The __src__ folder contains the implementation and training/evaluation utilities.

- __expressive_leaky_memory_neuron.py__: the implementation of the ELM model.
- __neuronio__: files related to visualising, training and evaluating on the NeuronIO dataset.
- __shd__: files related to downloading, training and evaluating on the Spiking Heidelberg Digits (SHD) dataset, and its SHD-Adding version.

Note: the PyTorch implementation seems to be about 2x slower than the jax version unfortunately.

## Dataset

Running the NeuronIO related code requires downloading the dataset first (~115GB).

- Download Train Data: [single-neurons-as-deep-nets-nmda-train-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data)
- Download Test Data (Data_test): [single-neurons-as-deep-nets-nmda-test-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data)
- For the original dataset source and reference implementation, see: [neuron_as_deep_net](https://github.com/SelfishGene/neuron_as_deep_net)

Running the SHD related code is possible without seperately downloading the dataset (~0.5GB).

- The small SHD daset will automaticall be downloaded upon running the related notebook.
- A dataloader for the introduced SHD-Adding dataset is provided in __/src/shd/shd_data_loader.py__
- For more information onf SHD, please checkout the following website: [spiking-heidelberg-datasets-shd](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

Running the LRA training/evaluation is not provided at the moment.

- To download the dataset, we recommend to checkout the following repository: [mega](https://github.com/facebookresearch/mega)
- For the input preprocessing, please refer to our preprint.

## Citation

If you like what you find, and use an ELM variant or the SHD-Adding dataset, please consider citing us:

[1] Spieler, A., Rahaman, N., Martius, G., Schölkopf, B., & Levina, A. (2023). The ELM Neuron: an Efficient and Expressive Cortical Neuron Model Can Solve Long-Horizon Tasks. arXiv preprint arXiv:2306.16922.

## ELM Neuron Implementation Summary

### Overview

The Expressive Leaky Memory (ELM) neuron is a biologically-inspired phenomenological model of cortical neurons designed to efficiently replicate the input-output relationship of detailed biophysical neuron models. The ELM architecture captures sophisticated cortical computations with significantly fewer parameters (thousands vs. millions) compared to previous approaches like Temporal Convolutional Networks.

### Key Paper Findings

#### Performance Achievements

- __ELM neuron__: 53K parameters achieving AUC ≈ 0.992 on NeuronIO dataset
- __Branch-ELM neuron__: 8K parameters (7× reduction) with similar performance
- Previous TCN approach required ~10M parameters for comparable accuracy
- Successfully solves long-range dependencies (e.g., Pathfinder-X with 16k context)

#### Architectural Requirements (from ablation studies)

1. __Memory units__: 10-20 memory-like hidden states required (far more than typical phenomenological models)
2. __Timescales__: Range of 1-150ms optimal, with 25ms being most crucial (matching cortical membrane timescales)
3. __Nonlinearity__: Highly nonlinear integration (MLP with at least 1 hidden layer) is essential
4. __Lambda (λ)__: Larger values (e.g., 5.0) enable rapid memory updates, critical for matching fast neuronal dynamics

### Core Architecture Components

The ELM neuron consists of four main components:

#### 1. __Synapse Dynamics__ (Current-based)

Models how synaptic input is filtered over time:

- __Synapse trace (s_t)__: Filtered version of input aiding coincidence detection
- __Synapse weights (w_s)__:
  - ELM v1: Fixed positive weights (not learned)
  - Branch-ELM v2: Learnable weights (crucial for branch-level gating)
- __Timescale (τ_s)__: Controls decay rate of synaptic traces

__Equation__:

```
s_t = κ_s * s_{t-1} + w_s * x_t
where κ_s = exp(-Δt / τ_s)
```

#### 2. __Integration Mechanism__ (Dendritic Processing)

How synaptic inputs are combined and processed:

__ELM v1__: Direct MLP processing of all synapses

```
Input to MLP: [s_t, κ_m * m_{t-1}]
```

__Branch-ELM v2__: Two-stage processing mimicking dendritic tree structure

```
1. Branch reduction: b_t = sum(w_s * x_t per branch)
2. MLP input: [b_t, κ_m * m_{t-1}]
```

- __MLP structure__: Typically 1 hidden layer with size 2*d_m
- __Activation__: ReLU or SiLU
- __Purpose__: Captures nonlinear dendritic computations

#### 3. __Memory Dynamics__ (Leaky Integration)

Core recurrent state maintaining temporal dependencies:

__ELM v1__:

```
Δm_t = tanh(MLP([syn_input, κ_m * m_{t-1}]))
m_t = κ_m * m_{t-1} + λ * (1 - κ_m) * Δm_t
```

__Branch-ELM v2__ (Modified for stability):

```
Δm_t = tanh(MLP([b_t, κ_m * m_{t-1}]))
m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t
where κ_λ = exp(-Δt * λ / τ_m)
```

__Key parameters__:

- __d_m__: Number of memory units (typically 100 for long-range tasks, 15-20 for NeuronIO)
- __τ_m__: Memory timescales (logspace from τ_min to τ_max, e.g., 1-150ms)
- __κ_m__: Decay factor = exp(-Δt / τ_m)
- __λ__: Scaling factor (typically 5.0)
  - v1: Scales the memory update magnitude
  - v2: Incorporated into κ_λ for improved stability

__Stability design__: The coupling of forget (κ_m) and input (1-κ_m or 1-κ_λ) timescales ensures bounded growth even with highly nonlinear MLPs, addressing vanishing/exploding gradients.

#### 4. __Output Dynamics__

Linear readout from memory state:

```
y_t = W_y * m_t
```

- __W_y__: Learnable output weights (d_o × d_m)
- Interpreted as spike probability when sigmoid applied

### Implementation Differences: ELM v1 vs Branch-ELM v2

#### ELM v1 (`expressive_leaky_memory_neuron.py`)

__State variables__:

- `s_t`: Synapse traces (shape: batch_size × num_synapse)
- `m_t`: Memory units (shape: batch_size × num_memory)

__Key characteristics__:

1. Processes all synaptic inputs directly through MLP
2. Fixed synapse weights (w_s initialized to 0.5, ReLU-activated)
3. Parameter `tau_s_value` for synapse timescales
4. Memory update: `m_t = κ_m * m_{t-1} + λ * (1 - κ_m) * Δm_t`

__Routing__: Can use `random_routing` or `neuronio_routing` to map inputs to synapses

__Pros__:

- Simpler architecture
- More flexible for arbitrary input structures

__Cons__:

- Higher parameter count (~53K for NeuronIO)
- Requires more synapses for complex tasks

#### Branch-ELM v2 (`expressive_leaky_memory_neuron_v2.py`)

__State variables__:

- `b_t`: Branch activations (shape: batch_size × num_branch)
- `m_t`: Memory units (shape: batch_size × num_memory)

__Key characteristics__:

1. Two-stage processing: synapse→branch→MLP
2. __Learnable synapse weights__ (w_s must be learned)
3. Parameter `tau_b_value` for branch timescales (replaces `tau_s`)
4. Modified memory update: `m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t`
5. Additional decay factor: `κ_λ = exp(-Δt * λ / τ_m)`

__Branch computation__:

```python
b_inp = (w_s * x_t).view(batch, num_branch, -1).sum(dim=-1)
b_t = κ_b * b_{t-1} + b_inp
```

__Routing__: Requires `neuronio_routing` for proper branch assignment

- Interlocks excitatory/inhibitory inputs
- Creates overlapping windows mapping inputs to branches
- Exploits locality in synaptic organization

__Pros__:

- 7× parameter reduction (~8K for NeuronIO)
- More stable training (especially with λ=5)
- Biologically plausible: mimics dendritic tree structure
- Better for inputs with spatial structure

__Cons__:

- Requires knowledge of input organization
- Must oversample (more synapses than inputs)
- w_s must be learnable (cannot be absorbed into MLP)

### Critical Implementation Details

#### Timescale Parameterization

Both versions use a reparameterization trick for bounded optimization:

```python
_proto_tau_m = inverse_scaled_sigmoid(
    torch.logspace(log10(tau_min), log10(tau_max), d_m),
    tau_min, tau_max
)
tau_m = scaled_sigmoid(_proto_tau_m, tau_min, tau_max)
```

This ensures τ_m stays within [tau_min, tau_max] while allowing gradient-based learning if enabled.

#### Routing Mechanisms

__random_routing__: Randomly selects num_synapse indices from num_input

__neuronio_routing__: Sophisticated assignment for biophysical neuron modeling

1. Interleaves excitatory/inhibitory inputs using `create_interlocking_indices`
2. Assigns neighboring inputs to same branch via overlapping windows
3. Creates validity mask for padding when inputs < synapses

#### Custom Activation

```python
custom_tanh(x) = tanh(x * 2/3) * 1.7159
```

Provides bounded activation with appropriate scaling for memory updates.

#### PyTorch JIT Compilation

Both implementations use `@jit.script_method` for:

- `dynamics()`: Single timestep computation
- `forward()`: Full sequence processing
- `neuronio_eval_forward()`: NeuronIO-specific inference
- `neuronio_viz_forward()`: Returns internal states for visualization

__Requirements for JIT__:

- Type annotations for all variables
- Use `torch.jit.annotate(List[torch.Tensor], [])` for list accumulation
- Avoid unsupported Python constructs (dict comprehensions, etc.)

### Hyperparameter Recommendations (from paper)

#### General

- __d_m__: Primary tuning parameter
  - NeuronIO: 15-20
  - SHD-Adding: 100
  - LRA tasks: 100-200
- __d_mlp__: If d_m is small, explore larger relative d_mlp (default: 2*d_m)
- __λ__: 5.0 for stability (especially Branch-ELM)
- __τ_m range__: Derived from task timescales
  - NeuronIO: 1-150ms
  - SHD: 1-1000ms
  - Learnability not strictly necessary if well-initialized

#### Task-Specific

__NeuronIO__ (fitting biophysical neuron):

- d_m = 20, d_mlp = 40, λ = 5.0
- τ_m: 1-150ms
- Branch-ELM: d_tree = 45, d_brch = 100

__SHD-Adding__ (long-range dependencies):

- d_m = 100, d_mlp = 200, λ = 5.0
- τ_m: 1-150ms (bounded to 1000ms)
- Longer τ_m crucial for performance

#### Training Stability

- Small d_m with many short τ_m or large λ can cause instability
- Branch-ELM v2 memory dynamics more stable
- Modified update rule (v2) resolves most stability issues

### Computational Efficiency

__FLOPs comparison__ (per sample inference on NeuronIO):

- TCN: ~10M parameters, high FLOPs
- LSTM: 266K parameters
- ELM v1: 53K parameters
- Branch-ELM v2: 8K parameters (lowest FLOPs)

__Speed note__: PyTorch implementation ~2× slower than JAX version

### Biological Interpretability

While individual parameters don't map directly to biophysical quantities, the architecture captures key computational principles:

1. __Synapse/Branch dynamics__: Filtering and local integration
2. __Memory timescales__: Multiple memory processes (calcium, voltage, adaptation)
3. __Nonlinear integration__: Dendritic nonlinearities (NMDA spikes, etc.)
4. __Leaky integration__: Membrane dynamics and long-term effects

The model demonstrates that:

- Multiple memory timescales (10-20) are necessary
- Timescales around 25ms are most critical (matches cortical membranes)
- Highly nonlinear integration is required (not simple summation)
- Rapid updates (large λ) needed to capture fast spike dynamics

### Dataset-Specific Features

#### NeuronIO

Both implementations include specialized methods:

- `neuronio_eval_forward()`: Applies sigmoid to spikes, scales soma voltage
- `neuronio_viz_forward()`: Returns predictions + internal states (s/b, m)
- Uses `DEFAULT_Y_TRAIN_SOMA_SCALE` constant for voltage scaling

#### SHD (Spiking Heidelberg Digits)

- Temporal binning of spike trains
- Default: 2ms bins
- Requires longer memory timescales for SHD-Adding variant

### Key Equations Summary

#### ELM v1

```
Synapse: s_t = κ_s * s_{t-1} + w_s * x_t
Branch:  b_t = sum(s_t per branch)
MLP:     Δm_t = tanh(MLP([b_t, κ_m * m_{t-1}]))
Memory:  m_t = κ_m * m_{t-1} + λ * (1 - κ_m) * Δm_t
Output:  y_t = W_y * m_t
```

#### Branch-ELM v2

```
Branch:  b_t = κ_b * b_{t-1} + sum(w_s * x_t per branch)
MLP:     Δm_t = tanh(MLP([b_t, κ_m * m_{t-1}]))
Memory:  m_t = κ_m * m_{t-1} + (1 - κ_λ) * Δm_t
Output:  y_t = W_y * m_t

where:
κ_s = exp(-Δt / τ_s)          (v1 only)
κ_b = exp(-Δt / τ_b)          (v2 only)
κ_m = exp(-Δt / τ_m)
κ_λ = exp(-Δt * λ / τ_m)      (v2 only)
```
