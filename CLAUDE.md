# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements the Expressive Leaky Memory (ELM) neuron, a biologically-inspired phenomenological model of cortical neurons that efficiently captures sophisticated neuronal computations. The implementation features two variants: ELM v1 and Branch-ELM v2, with v2 achieving ~7× parameter reduction through hierarchical dendritic processing.

## Code Quality and Formatting

This project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

**Standards enforced:**
- **black**: Code formatting (line length, style)
- **isort**: Import sorting (black-compatible profile)
- **pre-commit-hooks**: AST validation, YAML checks, trailing whitespace, end-of-file fixers

When modifying Python code, ensure it passes black and isort before committing.

## Environment Setup

This project is a pip-installable package. Choose one of the following installation methods:

### Method 1: Install from Source (Recommended for Development)

```bash
# Install in editable mode with all development tools
pip install -e ".[dev]"

# Or install with specific optional dependencies
pip install -e ".[wandb]"  # For experiment tracking
pip install -e ".[all]"    # All optional dependencies
```

### Method 2: Using Conda Environment

```bash
# Create conda environment (CPU or GPU version available)
conda env create -f elm_env.yml
# OR for GPU support:
conda env create -f elm_env_gpu.yml

# Activate environment
conda activate elm_env

# Then install the package
pip install -e .
```

### Method 3: Build and Install Wheel

```bash
# Build the wheel
python -m build --wheel

# Install the wheel
pip install dist/elmneuron-0.1.0-py3-none-any.whl
```

**Core Dependencies**: PyTorch >=2.0.0, NumPy, h5py, matplotlib, seaborn, scikit-learn, tqdm

**Optional Dependencies**: wandb (experiment tracking), pytest/black/isort (development)

See `INSTALL.md` for detailed installation instructions.

## Running Tests

```bash
# Run all tests
pytest tests/test_elm_models.py -v

# Run specific test class
pytest tests/test_elm_models.py::TestInitialization -v

# Run with detailed output
pytest tests/test_elm_models.py -v --tb=short
```

## Training and Evaluation

### NeuronIO Dataset

Training on NeuronIO requires downloading the dataset first (~115GB):
- Train: [single-neurons-as-deep-nets-nmda-train-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data)
- Test: [single-neurons-as-deep-nets-nmda-test-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data)

```bash
# Train ELM on NeuronIO
jupyter notebook notebooks/train_elm_on_neuronio.ipynb

# Or use the training script
python notebooks/neuronio_train_script.py

# Evaluate pre-trained models
jupyter notebook notebooks/eval_elm_on_neuronio.ipynb
```

### SHD Dataset

The Spiking Heidelberg Digits (SHD) dataset (~0.5GB) downloads automatically:

```bash
jupyter notebook notebooks/train_elm_on_shd.ipynb
```

## Architecture: Key Differences Between v1 and v2

### ELM v1 (`src/expressive_leaky_memory_neuron.py`)

**State variables**: `s_t` (synapse traces), `m_t` (memory)

**Processing flow**:
```
x_t → [synapse dynamics: s_t = κ_s·s_{t-1} + w_s·x_t] →
      [branch reduction: sum per branch] →
      [MLP([branch, κ_m·m_{t-1}])] → Δm_t →
      [memory: m_t = κ_m·m_{t-1} + λ·(1-κ_m)·Δm_t] →
      [output: y_t = W_y·m_t]
```

**Characteristics**:
- Fixed synapse weights (not typically learned)
- Processes all synaptic inputs directly
- Parameter count: ~53K for NeuronIO

### Branch-ELM v2 (`src/expressive_leaky_memory_neuron_v2.py`)

**State variables**: `b_t` (branch activations), `m_t` (memory)

**Processing flow**:
```
x_t → [weighted branch input: b_inp = sum(w_s·x_t per branch)] →
      [branch dynamics: b_t = κ_b·b_{t-1} + b_inp] →
      [MLP([b_t, κ_m·m_{t-1}])] → Δm_t →
      [modified memory: m_t = κ_m·m_{t-1} + (1-κ_λ)·Δm_t] →
      [output: y_t = W_y·m_t]
```

**Characteristics**:
- **Learnable synapse weights** (critical, cannot be absorbed into MLP)
- Two-stage hierarchical processing (synapse → branch → memory)
- Modified memory update with `κ_λ = exp(-Δt·λ/τ_m)` for improved stability
- Parameter count: ~8K for NeuronIO (7× reduction)
- Uses `tau_b_value` (branch timescale) instead of `tau_s_value` (synapse timescale)

**When to use which version**:
- Use v2 when input has spatial structure and you can apply `neuronio_routing`
- Use v1 for arbitrary input structures or when routing is not applicable
- v2 is more parameter-efficient and stable for complex tasks

## Critical Implementation Details

### PyTorch JIT Compilation

Both models use `@jit.script_method` decorators for performance. When modifying JIT-compiled methods:

1. **Type annotations are mandatory**: All variables must have explicit types
2. **List accumulation syntax**:
   ```python
   outputs = torch.jit.annotate(List[torch.Tensor], [])
   ```
3. **Avoid unsupported constructs**: Dict comprehensions, f-strings in some contexts, certain Python builtins

### Routing Mechanisms

**`neuronio_routing`**: Required for Branch-ELM v2, exploits spatial structure in inputs
- Interleaves excitatory/inhibitory inputs
- Creates overlapping windows mapping inputs to branches
- Must satisfy: `ceil(num_input / num_branch) <= num_synapse_per_branch`

**`random_routing`**: Random assignment of inputs to synapses
- More flexible but less biologically plausible
- Works with both v1 and v2

**No routing** (`None`): Direct 1-to-1 mapping
- Requires `num_synapse == num_input`

### Timescale Parameterization

Memory timescales use bounded optimization via scaled sigmoid:

```python
tau_m = scaled_sigmoid(_proto_tau_m, tau_min, tau_max)
```

This ensures τ_m stays within [tau_min, tau_max] while allowing gradient-based learning. The inverse transform is applied during initialization to start with logspace-distributed timescales.

### NeuronIO-Specific Methods

Both models provide specialized methods for NeuronIO dataset:

- **`neuronio_eval_forward()`**: Applies sigmoid to spike predictions, scales soma voltage
- **`neuronio_viz_forward()`**: Returns predictions + internal states (s_t/b_t, m_t) for visualization

## Hyperparameter Guidelines

### General Recommendations

| Parameter | NeuronIO | SHD-Adding | Description |
|-----------|----------|------------|-------------|
| `num_memory` (d_m) | 15-20 | 100 | Primary tuning parameter |
| `lambda_value` | 5.0 | 5.0 | Memory update scaling (critical for stability) |
| `memory_tau_min` | 1.0 ms | 1.0 ms | Minimum memory timescale |
| `memory_tau_max` | 150 ms | 1000 ms | Maximum memory timescale |
| `mlp_hidden_size` | 2×d_m | 2×d_m | MLP hidden dimension (default) |
| `mlp_num_layers` | 1 | 1 | Number of MLP hidden layers |
| `mlp_activation` | "relu" | "relu" | Activation function |

### Branch-ELM v2 Specific (NeuronIO)

```python
model = ELM_v2(
    num_input=1278,
    num_output=2,
    num_memory=15,
    lambda_value=5.0,
    num_branch=45,
    num_synapse_per_branch=100,
    input_to_synapse_routing='neuronio_routing',
    tau_b_value=5.0,
    memory_tau_min=1.0,
    memory_tau_max=150.0
)
```

### ELM v1 (NeuronIO)

```python
model = ELM_v1(
    num_input=1278,
    num_output=2,
    num_memory=20,
    lambda_value=5.0,
    tau_s_value=5.0,
    memory_tau_min=1.0,
    memory_tau_max=150.0
)
```

## Code Organization

The package is structured as an installable Python package named `elmneuron`:

```
src/elmneuron/                              # Main package (installed as 'elmneuron')
├── __init__.py
├── expressive_leaky_memory_neuron.py       # ELM v1 implementation
├── expressive_leaky_memory_neuron_v2.py    # Branch-ELM v2 implementation
├── modeling_utils.py                       # Shared utilities (MLP, activations, routing)
├── neuronio/                               # NeuronIO dataset utilities
│   ├── __init__.py
│   ├── neuronio_data_loader.py
│   ├── neuronio_data_utils.py
│   ├── neuronio_train_utils.py
│   ├── neuronio_eval_utils.py
│   └── neuronio_viz_utils.py
└── shd/                                    # SHD dataset utilities
    ├── __init__.py
    ├── shd_data_loader.py
    └── shd_download_utils.py

notebooks/
├── train_elm_on_neuronio.ipynb             # NeuronIO training notebook
├── eval_elm_on_neuronio.ipynb              # NeuronIO evaluation notebook
├── train_elm_on_shd.ipynb                  # SHD training notebook
└── neuronio_train_script.py                # Script version of training

tests/
└── test_elm_models.py                      # Comprehensive test suite

models/                                      # Pre-trained models (various sizes)
```

**Import paths:**
```python
# Correct imports (package is 'elmneuron', not 'src')
from elmneuron.expressive_leaky_memory_neuron import ELM as ELM_v1
from elmneuron.expressive_leaky_memory_neuron_v2 import ELM as ELM_v2
from elmneuron.modeling_utils import MLP, create_interlocking_indices
from elmneuron.neuronio.neuronio_data_loader import NeuronIODataset
from elmneuron.shd.shd_data_loader import SHDDataset
```

## Common Development Patterns

### Creating a Model Instance

Always check which version you're using and provide appropriate parameters:

```python
# v1: uses tau_s_value
from elmneuron.expressive_leaky_memory_neuron import ELM
model_v1 = ELM(num_input=128, num_output=2, num_memory=20, tau_s_value=5.0)

# v2: uses tau_b_value, w_s must be learnable
from elmneuron.expressive_leaky_memory_neuron_v2 import ELM
model_v2 = ELM(num_input=128, num_output=2, num_memory=20, tau_b_value=5.0)
```

### Training Loop Pattern

```python
model = ELM_v2(**config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)  # or model.neuronio_eval_forward(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
```

### Accessing Internal States for Debugging

```python
outputs, states, memory = model.neuronio_viz_forward(X)
# states: synapse traces (v1) or branch activations (v2)
# memory: memory unit activations over time
```

## Known Limitations

1. **PyTorch ~2× slower than JAX**: The original paper's JAX implementation is approximately twice as fast
2. **NeuronIO dataset size**: ~115GB, requires significant storage
3. **JIT compilation constraints**: Some Python features unavailable in JIT-compiled methods
4. **v2 routing requirement**: Branch-ELM v2 typically requires structured routing (neuronio_routing) to achieve best performance

## Model Files

Pre-trained models in `models/` directory are named by memory dimension (d_m):
- Smaller models (d_m=1-10): Faster but lower accuracy
- Optimal for NeuronIO: d_m=15-30
- Best effort model: d_m=100, AUC=0.9946

Load models with:
```python
model.load_state_dict(torch.load("models/elm_dm15.pt"))
```

## Git Workflow

**Main branch:** `main`

**Branch naming convention:**
- Feature branches: `feat/feature-name`
- Bug fixes: `fix/bug-description`
- Experiments: `exp/experiment-name`

**Commit practices:**
- Write clear, descriptive commit messages
- Reference issue numbers when applicable
- Ensure all tests pass before committing
- Pre-commit hooks will automatically format code

## Paper Reference

When making architectural changes, refer to the detailed implementation summary in README.md (lines 66-393) which provides comprehensive documentation of the mathematical formulations, design decisions, and biological interpretations.

## Package Building and Distribution

This repository is configured as a modern Python package using `pyproject.toml`:

**Building the package:**
```bash
# Build both wheel and source distribution
python -m build

# Build only wheel (faster)
python -m build --wheel

# Output: dist/elmneuron-0.1.0-py3-none-any.whl
```

**Package structure:**
- `pyproject.toml`: Modern package configuration (PEP 518, 621)
- `src/elmneuron/`: Source code in src-layout (recommended)
- `MANIFEST.in`: Includes documentation, models, and metadata
- `src/elmneuron/py.typed`: PEP 561 type hint marker

**Version bumping:**
Update version in `pyproject.toml` and `src/elmneuron/__init__.py`

**Publishing to PyPI** (when ready):
```bash
python -m build
python -m twine upload dist/*
```

## Additional Notes

**Working with internal model states:**
Both ELM versions provide specialized methods for accessing internal states:
- Use `neuronio_viz_forward()` instead of `forward()` when you need to inspect intermediate activations
- This returns `(outputs, states, memory)` where states are synapse/branch activations and memory is the memory unit activations
- Essential for debugging, visualization, and analysis tasks

**Modifying JIT-compiled code:**
If you encounter errors when modifying model code, check:
1. All variables have type annotations
2. No f-strings in JIT contexts (use `"string {}".format()` instead)
3. List accumulation uses `torch.jit.annotate(List[torch.Tensor], [])`
4. No dict comprehensions or unsupported Python constructs

**Import paths:**
Always use `from elmneuron import ...` not `from src.elmneuron import ...`
The package name is `elmneuron` when installed.
