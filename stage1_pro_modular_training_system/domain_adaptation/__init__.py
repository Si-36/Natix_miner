"""
Domain adaptation module for Phase 4+

ExPLoRA (Extended Parameter-Efficient Low-Rank Adaptation) for domain adaptation.
"""

from .explora import ExPLoRATrainer
from .data import UnlabeledRoadDataset

__all__ = ["ExPLoRATrainer", "UnlabeledRoadDataset"]
