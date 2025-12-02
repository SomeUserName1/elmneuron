"""
NeuronIO Dataset for PyTorch Lightning.

Simplified IterableDataset that leverages DataLoader's built-in
multiprocessing, prefetching, and pin_memory features.
"""

from typing import List

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .neuronio_data_utils import (
    DEFAULT_Y_SOMA_THRESHOLD,
    DEFAULT_Y_TRAIN_SOMA_BIAS,
    DEFAULT_Y_TRAIN_SOMA_SCALE,
    NEURONIO_DATA_DIM,
    NEURONIO_LABEL_DIM,
    NEURONIO_SIM_LEN,
    NEURONIO_SIM_PER_FILE,
    create_neuronio_input_type,
    parse_sim_experiment_file,
)


def preprocess_data(
    X,
    y_spike,
    y_soma,
    y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
):
    """Preprocess raw data from HDF5 file."""
    # Convert to torch tensors
    X = torch.from_numpy(X).float().permute(2, 1, 0)
    y_spike = torch.from_numpy(y_spike).float().T.unsqueeze(2)
    y_soma = torch.from_numpy(y_soma).float().T.unsqueeze(2)

    # Apply thresholding
    y_soma[y_soma > y_soma_threshold] = y_soma_threshold

    # Bias correction and scaling
    y_soma = (y_soma - y_train_soma_bias) * y_train_soma_scale

    return X, y_spike, y_soma


class NeuronIO(IterableDataset):
    """
    NeuronIO IterableDataset optimized for PyTorch Lightning.

    This dataset leverages DataLoader's built-in features:
    - num_workers for parallel loading
    - pin_memory for faster GPU transfer
    - prefetch_factor for prefetching

    Each worker automatically gets a subset of files via get_worker_info().
    """

    def __init__(
        self,
        batches_per_epoch: int,
        file_paths: List[str],
        synapse_types=None,
        batch_size: int = 8,
        input_window_size: int = 500,
        ignore_time_from_start: int = 150,
        y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
        y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
        y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
        neuronio_sim_per_file: int = NEURONIO_SIM_PER_FILE,
        neuronio_sim_len: int = NEURONIO_SIM_LEN,
        neuronio_label_dim: int = NEURONIO_LABEL_DIM,
        neuronio_data_dim: int = NEURONIO_DATA_DIM,
        seed: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize NeuronIO dataset.

        Args:
            batches_per_epoch: Number of batches to yield per epoch
            file_paths: List of HDF5 file paths
            synapse_types: Excitatory/inhibitory markers (possibly routed)
            batch_size: Samples per batch
            input_window_size: Temporal window size
            file_load_fraction: Fraction of batches to generate per file
            ignore_time_from_start: Skip initial timesteps
            y_soma_threshold: Soma voltage threshold
            y_train_soma_bias: Soma voltage bias
            y_train_soma_scale: Soma voltage scale
            neuronio_sim_per_file: Simulations per file
            neuronio_sim_len: Simulation length
            neuronio_label_dim: Label dimension
            neuronio_data_dim: Data dimension
            seed: Random seed
            verbose: Enable verbose logging
        """
        super().__init__()
        self.batches_per_epoch = batches_per_epoch
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.ignore_time_from_start = ignore_time_from_start
        self.y_soma_threshold = y_soma_threshold
        self.y_train_soma_bias = y_train_soma_bias
        self.y_train_soma_scale = y_train_soma_scale
        self.neuronio_sim_per_file = neuronio_sim_per_file
        self.neuronio_sim_len = neuronio_sim_len
        self.neuronio_label_dim = neuronio_label_dim
        self.neuronio_data_dim = neuronio_data_dim
        self.seed = seed
        self.verbose = verbose

        # Setup synapse types
        if synapse_types is None:
            synapse_types = create_neuronio_input_type()
        self.synapse_types = torch.tensor(synapse_types, dtype=torch.float32)

        # Calculate batches per file
        self.batches_per_file = int(
                    (neuronio_sim_per_file * neuronio_sim_len) / (
            batch_size * input_window_size
        ))

    def __iter__(self):
        """Yield batches, distributing files across DataLoader workers."""
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process mode
            files = self.file_paths
            worker_seed = self.seed
        else:
            # Multi-worker mode: distribute files across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Interleaved distribution for better load balancing
            files = self.file_paths[worker_id::num_workers]
            worker_seed = self.seed + worker_id

        # Initialize RNGs
        file_rng = np.random.default_rng(worker_seed)
        batch_rng = np.random.default_rng(worker_seed + 1000)

        # Track batches yielded
        batches_yielded = 0

        while batches_yielded < self.batches_per_epoch:
            # Randomly select a file
            file_path = file_rng.choice(files)

            if self.verbose:
                print(f"Loading file: {file_path}", flush=True)

            # Load and preprocess data
            X, y_spike, y_soma = parse_sim_experiment_file(
                sim_experiment_file=file_path,
                include_params=False,
                verbose=self.verbose,
            )

            X, y_spike, y_soma = preprocess_data(
                X=X,
                y_spike=y_spike,
                y_soma=y_soma,
                y_soma_threshold=self.y_soma_threshold,
                y_train_soma_bias=self.y_train_soma_bias,
                y_train_soma_scale=self.y_train_soma_scale,
            )

            # Generate batches from this file
            for batch in self._generate_batches(X, y_spike, y_soma, batch_rng):
                yield batch
                batches_yielded += 1
                if batches_yielded >= self.batches_per_epoch:
                    break

    def _generate_batches(self, X, y_spike, y_soma, rng):
        """Generate batches from loaded file data."""
        for _ in range(self.batches_per_file):
            # Randomly sample simulations for current batch
            selected_sim_inds = rng.choice(
                self.neuronio_sim_per_file, size=self.batch_size, replace=False
            )

            # Randomly sample timepoints for current batch
            max_start = (
                self.neuronio_sim_len
                - self.input_window_size
                - self.ignore_time_from_start
            )
            selected_time_inds = (
                rng.integers(0, max_start, size=self.batch_size)
                + self.ignore_time_from_start
            )

            # Initialize batch tensors
            X_batch = torch.zeros(
                (self.batch_size, self.input_window_size, self.neuronio_data_dim)
            )
            y_spike_batch = torch.zeros(
                (self.batch_size, self.input_window_size, self.neuronio_label_dim // 2)
            )
            y_soma_batch = torch.zeros(
                (self.batch_size, self.input_window_size, self.neuronio_label_dim // 2)
            )

            # Gather batch data
            for k, (sim_ind, time_ind) in enumerate(
                zip(selected_sim_inds, selected_time_inds)
            ):
                X_batch[k] = X[sim_ind, time_ind : time_ind + self.input_window_size, :]
                y_spike_batch[k] = y_spike[
                    sim_ind, time_ind : time_ind + self.input_window_size, :
                ]
                y_soma_batch[k] = y_soma[
                    sim_ind, time_ind : time_ind + self.input_window_size, :
                ]

            # Apply synapse types (excitatory/inhibitory markers)
            X_batch = X_batch * self.synapse_types

            # Squeeze label dimensions
            y_spike_batch = y_spike_batch.squeeze(-1)
            y_soma_batch = y_soma_batch.squeeze(-1)

            yield X_batch, (y_spike_batch, y_soma_batch)
