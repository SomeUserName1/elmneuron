"""
NeuronIO Training Script - Lightning Version

This script trains an ELM model on the NeuronIO dataset using PyTorch Lightning.
It uses the new Lightning-based architecture with DataModules, LightningModules,
and callbacks for visualization.
"""

import json
import os
import random
import tempfile
from pathlib import Path

import kagglehub
import numpy as np
import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from elmneuron.callbacks import (
    MemoryDynamicsCallback,
    SequenceVisualizationCallback,
    StateRecorderCallback,
)
from elmneuron.expressive_leaky_memory_neuron_v2 import ELM
from elmneuron.neuronio.neuronio_data_utils import (
    NEURONIO_DATA_DIM,
    NEURONIO_LABEL_DIM,
)
from elmneuron.neuronio.neuronio_datamodule import NeuronIODataModule
from elmneuron.tasks.neuronio_task import NeuronIOTask
from elmneuron.transforms import NeuronIORouting

if __name__ == "__main__":
    # ######### Logging Config ##########
    print("Wandb configuration started...")

    # Setup directory for saving training artifacts
    temporary_dir = tempfile.TemporaryDirectory()
    artifacts_dir = Path(temporary_dir.name) / "training_artifacts"
    os.makedirs(str(artifacts_dir))

    # Wandb config
    entity_name = "someusername1"
    project_name = "elm-neuron-analysis"

    # Login to wandb
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    # ######### General Config ##########
    print("General configuration started...")

    # General Config
    general_config = dict()
    general_config["seed"] = 0
    general_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    general_config["short_training_run"] = False
    general_config["verbose"] = True  # Enable verbose logging for debugging
    print("Device:", general_config["device"])

    # Seeding & Determinism
    os.environ["PYTHONHASHSEED"] = str(general_config["seed"])
    random.seed(general_config["seed"])
    np.random.seed(general_config["seed"])
    torch.manual_seed(general_config["seed"])
    torch.cuda.manual_seed(general_config["seed"])
    torch.backends.cudnn.deterministic = True

    # Set float32 matmul precision for better performance on Tensor Cores
    torch.set_float32_matmul_precision("high")

    # ######### Data, Model and Training Config ##########
    print("Data, model and training configuration started...")

    # Download datasets from Kaggle
    # Train: https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data
    # Test: https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data

    train_data_dir_path = Path(
        kagglehub.dataset_download(
            "selfishgene/single-neurons-as-deep-nets-nmda-train-data"
        )
    )
    test_data_dir_path = Path(
        kagglehub.dataset_download(
            "selfishgene/single-neurons-as-deep-nets-nmda-test-data"
        )
    )

    # Data Config
    data_config = dict()
    train_data_dirs = [
        str(train_data_dir_path / "full_ergodic_train_batch_2"),
        str(train_data_dir_path / "full_ergodic_train_batch_3"),
        str(train_data_dir_path / "full_ergodic_train_batch_4"),
        str(train_data_dir_path / "full_ergodic_train_batch_5"),
        str(train_data_dir_path / "full_ergodic_train_batch_6"),
        str(train_data_dir_path / "full_ergodic_train_batch_7"),
        str(train_data_dir_path / "full_ergodic_train_batch_8"),
        str(train_data_dir_path / "full_ergodic_train_batch_9"),
        str(train_data_dir_path / "full_ergodic_train_batch_10"),
    ]
    valid_data_dirs = [str(train_data_dir_path / "full_ergodic_train_batch_1")]
    test_data_dirs = [str(test_data_dir_path)]

    data_config["train_data_dirs"] = train_data_dirs
    data_config["valid_data_dirs"] = valid_data_dirs
    data_config["test_data_dirs"] = test_data_dirs
    data_config["data_dim"] = NEURONIO_DATA_DIM
    data_config["label_dim"] = NEURONIO_LABEL_DIM

    # Model Config
    model_config = dict()
    model_config["num_branch"] = 45
    model_config["num_synapse_per_branch"] = 100
    model_config["num_memory"] = 20
    model_config["memory_tau_min"] = 1.0
    model_config["memory_tau_max"] = 1000.0
    model_config["learn_memory_tau"] = False
    model_config["lambda_value"] = 5.0
    model_config["tau_b_value"] = 5.0

    # Training Config
    train_config = dict()
    train_config["num_epochs"] = 5 if general_config["short_training_run"] else 35
    train_config["learning_rate"] = 5e-4
    train_config["batch_size"] = 32 if general_config["short_training_run"] else 8
    train_config["batches_per_epoch"] = (
        1000 if general_config["short_training_run"] else 10000
    )
    train_config["batches_per_epoch"] = int(
        8 / train_config["batch_size"] * train_config["batches_per_epoch"]
    )
    train_config["file_load_fraction"] = (
        0.5 if general_config["short_training_run"] else 0.3
    )
    train_config["num_prefetch_batch"] = 20
    train_config["num_workers"] = 5
    train_config["input_window_size"] = 500

    # Save Configs
    with open(str(artifacts_dir / "general_config.json"), "w", encoding="utf-8") as f:
        json.dump(general_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artifacts_dir / "data_config.json"), "w", encoding="utf-8") as f:
        json.dump(data_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artifacts_dir / "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artifacts_dir / "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=4, sort_keys=True)

    # ######### Setup Lightning Components ##########
    print("Setting up Lightning components...")

    # Create routing transform
    routing = NeuronIORouting(
        num_input=NEURONIO_DATA_DIM,
        num_branch=model_config["num_branch"],
        num_synapse_per_branch=model_config["num_synapse_per_branch"],
    )

    # Calculate total number of synapses after routing
    num_synapse = routing.num_synapse

    # Create DataModule
    from elmneuron.neuronio.neuronio_data_utils import (
        get_data_files_from_folder,
    )

    print("Loading data files...")
    train_files = get_data_files_from_folder(data_config["train_data_dirs"])
    print(f"  Train files: {len(train_files)} files found")
    valid_files = get_data_files_from_folder(data_config["valid_data_dirs"])
    print(f"  Valid files: {len(valid_files)} files found")
    test_files = get_data_files_from_folder(data_config["test_data_dirs"])
    print(f"  Test files: {len(test_files)} files found")

    print("Creating NeuronIODataModule...")
    datamodule = NeuronIODataModule(
        train_files=train_files,
        val_files=valid_files,
        test_files=test_files,
        routing=routing,
        batch_size=train_config["batch_size"],
        input_window_size=train_config["input_window_size"],
        file_load_fraction=train_config["file_load_fraction"],
        num_workers=train_config["num_workers"],
        num_prefetch_batch=train_config["num_prefetch_batch"],
        train_batches_per_epoch=train_config["batches_per_epoch"],
        val_batches_per_epoch=train_config["batches_per_epoch"] // 10,
        test_batches_per_epoch=train_config["batches_per_epoch"] // 5,
        seed=general_config["seed"],
        verbose=general_config["verbose"],
    )
    print("NeuronIODataModule created successfully")

    # Create base ELM model with hierarchical branch structure
    # Note: compile_mode=None disables torch.compile as it causes loop unrolling
    # for RNN-style models with Python loops, leading to massive graphs
    elm_model = ELM(
        num_input=num_synapse,  # After routing
        num_output=NEURONIO_LABEL_DIM,
        num_memory=model_config["num_memory"],
        num_branch=model_config["num_branch"],
        num_synapse_per_branch=model_config["num_synapse_per_branch"],
        lambda_value=model_config["lambda_value"],
        tau_b_value=model_config["tau_b_value"],
        memory_tau_min=model_config["memory_tau_min"],
        memory_tau_max=model_config["memory_tau_max"],
        learn_memory_tau=model_config["learn_memory_tau"],
        compile_mode=None,  # Disable compilation to avoid loop unrolling
    )

    # Wrap in Lightning task module
    lightning_module = NeuronIOTask(
        model=elm_model,
        learning_rate=train_config["learning_rate"],
        optimizer="adam",
        scheduler="cosine",
        scheduler_kwargs={
            "T_max": train_config["num_epochs"] * train_config["batches_per_epoch"]
        },
    )

    print(
        f"Model initialized with {sum(p.numel() for p in elm_model.parameters())} parameters"
    )

    # ######### Setup Callbacks ##########
    print("Setting up callbacks...")

    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath=str(artifacts_dir / "checkpoints"),
            filename="elm-{epoch:02d}-{val/spike_auc:.4f}",
            monitor="val/spike_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val/spike_auc",
            patience=10,
            mode="max",
            verbose=True,
        ),
        # Visualization callbacks
        StateRecorderCallback(
            record_every_n_epochs=5,
            num_samples=8,
            save_dir=str(artifacts_dir / "states"),
        ),
        SequenceVisualizationCallback(
            log_every_n_epochs=5,
            num_samples=4,
            task_type="regression",  # For soma voltage
            save_dir=str(artifacts_dir / "visualizations"),
            log_to_wandb=True,
        ),
        MemoryDynamicsCallback(
            log_every_n_epochs=5,
            num_samples=2,
            save_dir=str(artifacts_dir / "memory"),
            log_to_wandb=True,
        ),
    ]

    # ######### Setup Trainer ##########
    print("Setting up trainer...")

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=project_name,
        entity=entity_name,
        save_dir=str(artifacts_dir),
        log_model=True,
    )

    # Log all configs to wandb
    wandb_logger.experiment.config.update(
        {
            "general": general_config,
            "data": data_config,
            "model": model_config,
            "training": train_config,
        }
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=train_config["num_epochs"],
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        deterministic=True,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # ######### Training ##########
    print("Training started...", flush=True)
    print("  Calling datamodule.setup()...", flush=True)
    datamodule.setup("fit")
    print("  DataModule setup complete", flush=True)
    print(
        f"  Train dataloader batches: {len(datamodule.train_dataloader())}", flush=True
    )
    print(f"  Val dataloader batches: {len(datamodule.val_dataloader())}", flush=True)
    print("  Starting trainer.fit()...", flush=True)

    trainer.fit(lightning_module, datamodule=datamodule)

    # ######### Testing ##########
    print("Testing started...")

    # Test with best checkpoint
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")

    # ######### Save Final Model ##########
    print("Saving final model...")

    # Save best model state dict
    best_model_path = str(artifacts_dir / "neuronio_best_model_state.pt")
    torch.save(lightning_module.model.state_dict(), best_model_path)

    # Copy artifacts to local directory
    import shutil

    local_dir = Path(os.getcwd()) / "artifacts"
    if not local_dir.exists():
        os.mkdir(local_dir)

    # Copy entire artifacts directory
    if local_dir.exists():
        shutil.rmtree(local_dir)
    shutil.copytree(artifacts_dir, local_dir)

    print(f"Artifacts saved to: {local_dir}")

    # ######### Finish ##########
    wandb.finish()
    temporary_dir.cleanup()

    print("Training completed successfully!")
