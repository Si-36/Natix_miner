"""
Scheduler creation preserving exact lr_lambda logic from train_stage1_head.py

Cosine annealing + warmup scheduler with exact implementation.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def create_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int
) -> LambdaLR:
    """
    Create cosine annealing + warmup scheduler preserving EXACT lr_lambda logic.
    
    Preserves exact implementation from train_stage1_head.py:
    ```python
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    ```
    
    Args:
        optimizer: Optimizer instance
        total_steps: Total training steps
        warmup_steps: Warmup steps
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)
