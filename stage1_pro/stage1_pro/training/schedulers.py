import math
import torch.optim as optim
from typing import Any


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Any,
    num_training_steps: int,
):
    """
    Build cosine annealing scheduler with warmup.

    Exact match to baseline train_stage1_head.py:
    - Linear warmup for warmup_steps
    - Cosine annealing for remaining steps
    """
    warmup_steps = config.warmup_epochs * (num_training_steps // config.epochs)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
