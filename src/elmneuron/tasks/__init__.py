"""
Task-specific LightningModule wrappers for ELM models.

This module provides task-specific training logic, loss functions,
and metrics for different downstream tasks.
"""

from .classification_task import ClassificationTask
from .neuronio_task import NeuronIOTask
from .regression_task import RegressionTask

__all__ = [
    "NeuronIOTask",
    "ClassificationTask",
    "RegressionTask",
]
