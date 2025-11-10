"""
Long Range Arena (LRA) dataset module.

Provides DataModules for the LRA benchmark tasks.
"""

from .lra_datamodule import (
    ListOpsDataModule,
    LRAImageDataModule,
    LRAPathfinderDataModule,
    LRARetrievalDataModule,
    LRATextDataModule,
)

__all__ = [
    "ListOpsDataModule",
    "LRATextDataModule",
    "LRARetrievalDataModule",
    "LRAImageDataModule",
    "LRAPathfinderDataModule",
]
