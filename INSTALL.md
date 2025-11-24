# Installation Guide

This guide explains how to install the `elmneuron` package.

## Installation Methods

### Method 1: Install from source (Development)

For development or to get the latest code:

```bash
# Clone the repository
git clone git@github.com:AaronSpieler/elmneuron.git
cd elmneuron

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Method 2: Install from source (User)

For regular installation from source:

```bash
# Clone the repository
git clone git@github.com:AaronSpieler/elmneuron.git
cd elmneuron

# Install the package
pip install .

# Or with optional dependencies
pip install ".[wandb]"  # For Weights & Biases logging
```

### Method 3: Using Conda environment

Use the provided conda environment files:

```bash
# For CPU-only installation
conda env create -f elm_env.yml
conda activate elm_env

# For GPU support
conda env create -f elm_env_gpu.yml
conda activate elm_env
```

Then install the package in editable mode:

```bash
pip install -e .
```

## Optional Dependencies

The package supports several optional dependency groups:

- `dev`: Development tools (pytest, black, isort, pre-commit)
- `wandb`: Weights & Biases integration for experiment tracking
- `gpu`: GPU-enabled PyTorch (requires manual installation)
- `all`: All optional dependencies (except GPU)

Install with:

```bash
pip install ".[dev,wandb]"
```

## GPU Support

For GPU support, you need to install PyTorch with CUDA support separately. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions specific to your CUDA version.

Example for CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install .
```

## Verifying Installation

Test your installation:

```python
import elmneuron
from elmneuron import ELM_v1, ELM_v2

print(f"elmneuron version: {elmneuron.__version__}")

# Create a simple model
model = ELM_v2(
    num_input=128,
    num_output=2,
    num_memory=15,
)
print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
```

## Development Installation

For development with pre-commit hooks:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Troubleshooting

### Import errors

If you get import errors, make sure the package is properly installed:

```bash
pip list | grep elmneuron
```

### PyTorch compatibility

The package requires PyTorch >= 2.0.0. If you have an older version:

```bash
pip install --upgrade torch
```

### Pre-trained models

Pre-trained models are included in the `models/` directory. To use them:

```python
import torch
from elmneuron import ELM_v2

model = ELM_v2(num_input=1278, num_output=2, num_memory=15)
model.load_state_dict(torch.load("models/elm_dm15.pt"))
```

## Uninstallation

To uninstall the package:

```bash
pip uninstall elmneuron
```
