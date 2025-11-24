"""
NeuronIO dataset utilities for the ELM neuron.

This module provides data loading, training, evaluation, and visualization
utilities specifically for the NeuronIO dataset, which contains biophysical
neuron simulation data.
"""

from elmneuron.neuronio.neuronio_data_loader import NeuronIO, preprocess_data
from elmneuron.neuronio.neuronio_data_utils import (
    create_neuronio_input_type,
    visualize_training_batch,
)
from elmneuron.neuronio.neuronio_eval_utils import (
    NeuronioEvaluator,
    compute_test_predictions,
    get_num_trainable_params,
)
from elmneuron.neuronio.neuronio_train_utils import NeuronioLoss
from elmneuron.neuronio.neuronio_viz_utils import visualize_neuron_workings

__all__ = [
    # Data loading
    "NeuronIO",
    "preprocess_data",
    # Data utilities
    "create_neuronio_input_type",
    "visualize_training_batch",
    # Evaluation
    "NeuronioEvaluator",
    "compute_test_predictions",
    "get_num_trainable_params",
    # Training
    "NeuronioLoss",
    # Visualization
    "visualize_neuron_workings",
]
