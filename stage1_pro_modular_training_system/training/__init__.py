"""
Training module for Stage-1 Pro Modular Training System

Provides losses, optimizers, schedulers, EMA, trainer, and REAL PEFT training (Phase 4.7).
"""

from .losses import SelectiveLoss, AuxiliaryLoss, ConformalRiskLoss
from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .ema import EMA
from .trainer import Stage1ProTrainer
from .peft_real_trainer import RealPEFTTrainer, create_real_peft_trainer

__all__ = [
    "SelectiveLoss",
    "AuxiliaryLoss",
    "ConformalRiskLoss",
    "create_optimizer",
    "create_scheduler",
    "EMA",
    "Stage1Trainer",
    "RealPEFTTrainer",  # REAL library usage (Phase 4.7)
    "create_real_peft_trainer",
]
