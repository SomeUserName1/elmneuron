# Installation Guide

This guide explains how to install the `elmneuron` package with PyTorch Lightning support.

## Quick Start

```bash
# Clone and install in editable mode with all dependencies
git clone git@github.com:AaronSpieler/elmneuron.git
cd elmneuron
pip install -e ".[all]"
```

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

### Method 3: Build and install from wheel

Build a distributable wheel package:

```bash
# Install build tools if not already installed
pip install build

# Build the wheel (creates dist/elmneuron-0.1.0-py3-none-any.whl)
python -m build --wheel

# Install the wheel
pip install dist/elmneuron-0.1.0-py3-none-any.whl

# Or install with optional dependencies
pip install "dist/elmneuron-0.1.0-py3-none-any.whl[wandb]"
```

To build both wheel and source distribution:

```bash
python -m build
```

### Method 4: Using venv with requirements.txt

Create a virtual environment and install from requirements:

```bash
# Create virtual environment
python -m venv elm_env

# Activate the environment
# On Linux/macOS:
source elm_env/bin/activate
# On Windows:
# elm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Or install from wheel after building
python -m build --wheel
pip install dist/elmneuron-0.1.0-py3-none-any.whl
```

### Method 5: Using Conda environment

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

## Core Dependencies

The package requires:
- **PyTorch >= 2.0.0**: Deep learning framework
- **PyTorch Lightning >= 2.0.0**: Training framework
- **NumPy**: Numerical computing
- **h5py**: HDF5 file handling
- **torchvision, torchtext**: Dataset utilities
- **matplotlib, seaborn**: Visualization
- **scikit-learn**: Metrics and utilities
- **tqdm**: Progress bars

## Optional Dependencies

The package supports several optional dependency groups:

- `dev`: Development tools (pytest, black, isort, pre-commit, flake8)
- `wandb`: Weights & Biases integration for experiment tracking
- `tensorboard`: TensorBoard logging
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
from elmneuron.expressive_leaky_memory_neuron_v2 import ELM
from elmneuron.tasks.classification_task import ClassificationTask
from elmneuron.vision.vision_datamodule import MNISTDataModule
from lightning.pytorch import Trainer

print(f"elmneuron version: {elmneuron.__version__}")

# Create a simple model
model = ELM(
    num_input=784,
    num_output=10,
    num_memory=50,
    lambda_value=5.0,
)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test Lightning integration
task = ClassificationTask(model=model, learning_rate=1e-3)
print(f"✓ Lightning task module created")

# Test DataModule
datamodule = MNISTDataModule(data_dir="./data", batch_size=64)
print(f"✓ DataModule created")

print("\n✓ Installation verified successfully!")
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
