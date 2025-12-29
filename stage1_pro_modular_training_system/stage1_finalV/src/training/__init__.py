"""
Training module for Stage-1 Pro System (2025 Best Practices)
"""

from models.backbone import DINOv3Backbone
from .lightning_module import Phase1LightningModule, Phase1LightningModuleConfig, TrainingMetrics
from .lightning_data_module import RoadworkDataModule, RoadworkDataset

__all__ = [
    "Phase1LightningModule",
    "Phase1LightningModuleConfig",
    "TrainingMetrics",
    "RoadworkDataModule",
    "RoadworkDataset",
]
