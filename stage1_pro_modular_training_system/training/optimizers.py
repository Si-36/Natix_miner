"""
Optimizer creation preserving exact config from train_stage1_head.py

Phase 1-4: AdamW (exact config)
Phase 5+: F-SAM implementation (CVPR 2024)
"""

import torch
from torch.optim import AdamW
from typing import Optional


def create_optimizer(
    model,
    optimizer_type: str = "adamw",
    lr_head: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 0.01,
    phase: int = 1
):
    """
    Create optimizer preserving exact config from train_stage1_head.py.
    
    Phase 1-4: AdamW with exact config (betas=(0.9,0.999), eps=1e-8, weight_decay)
    Phase 5+: F-SAM implementation (CVPR 2024)
    
    Args:
        model: Model to optimize
        optimizer_type: "adamw" or "fsam"
        lr_head: Learning rate for head
        lr_backbone: Learning rate for backbone (if PEFT enabled)
        weight_decay: Weight decay
        phase: Phase number
    
    Returns:
        Optimizer instance
    """
    if optimizer_type == "adamw" or phase < 5:
        # Preserve exact config from train_stage1_head.py
        # AdamW with betas=(0.9,0.999), eps=1e-8, weight_decay
        return AdamW(
            model.parameters(),
            lr=lr_head,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    elif optimizer_type == "fsam" and phase >= 5:
        # TODO: Implement F-SAM (Phase 5)
        # F-SAM: Friendly SAM (CVPR 2024)
        # Two-step optimizer (forward + backward)
        # Adversarial perturbation formation
        # Gradient checkpointing support
        raise NotImplementedError("F-SAM implementation - Phase 5 only")
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type} for phase {phase}")


def create_per_layer_optimizer(
    model,
    transformer_params,
    head_params,
    lr_transformer: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 0.01
):
    """
    Create optimizer with per-layer LR adaptation (Phase 4+).
    
    Lower LR for transformer (if PEFT enabled)
    Higher LR for head
    
    Args:
        model: Model
        transformer_params: Transformer parameters (if PEFT enabled)
        head_params: Head parameters
        lr_transformer: LR for transformer
        lr_head: LR for head
        weight_decay: Weight decay
    
    Returns:
        Optimizer with param_groups
    """
    param_groups = [
        {"params": transformer_params, "lr": lr_transformer, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
    ]
    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
