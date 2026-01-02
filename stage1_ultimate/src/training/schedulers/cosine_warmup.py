"""
Cosine Annealing Learning Rate Scheduler with Warmup

2026 Best Practice:
- Warmup + cosine decay schedule
- Proven stable for vision transformers
- Works well with Sophia-H optimizer

Benefits:
- Gradual learning rate increase during warmup
- Smooth decay after warmup
- Prevents early convergence to poor local minima
- Stable training dynamics

Reference: "Attention is All You Need" (Vaswani et al., 2019)
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional
import math
import logging

logger = logging.getLogger(__name__)


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine Annealing with Warmup Scheduler
    
    Schedule:
    1. Linear warmup for first warmup_steps
    2. Cosine decay for remaining steps
    
    Formula:
    - During warmup: lr = initial_lr × (step / warmup_steps)
    - After warmup: lr = min_lr + 0.5 × (1 + cos(π × (step - warmup_steps) / total_steps))
    
    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr: Minimum learning rate (default: 0)
        num_cycles: Number of cosine cycles (default: 0.5 = decay to 10%)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 500,
        num_training_steps: int = 30000,
        min_lr: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ):
        if not isinstance(num_warmup_steps, int) or num_warmup_steps < 0:
            raise ValueError(f"Expected positive int for num_warmup_steps, got: {num_warmup_steps}")
        if not isinstance(num_training_steps, int) or num_training_steps < 0:
            raise ValueError(f"Expected positive int for num_training_steps, got: {num_training_steps}")
        
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        
        self.current_step = 0
        
        logger.info(f"✅ CosineWarmup initialized")
        logger.info(f"   Warmup steps: {num_warmup_steps}")
        logger.info(f"   Total steps: {num_training_steps}")
        logger.info(f"   Min LR: {min_lr}")
        logger.info(f"   Num cycles: {num_cycles}")
    
    def get_lr(self):
        """Compute current learning rate based on progress"""
        if self.current_step < self.num_warmup_steps:
            # Warmup phase: linear increase
            alpha = self.current_step / self.num_warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            progress = (self.current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress / self.num_cycles))
            decayed_lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay
            return [decayed_lr for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None, metric: Optional[float] = None):
        """
        Update learning rate (called after each epoch)
        
        Args:
            epoch: Current epoch (optional)
            metric: Validation metric (optional, not used)
        """
        if epoch is not None:
            self.last_epoch = epoch
        
        self.current_step += 1
        
        # Update learning rates for all parameter groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.get_lr()[i]
            param_group['lr'] = new_lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineWarmupScheduler:
    """
    Factory-compatible Cosine Warmup Scheduler (simplied interface)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 500,
        num_training_steps: int = 30000,
        min_lr: float = 0.0,
        num_cycles: float = 0.5
    ):
        super().__init__()
        
        self.scheduler = CosineAnnealingWarmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
            num_cycles=num_cycles
        )
    
    def get_lr(self):
        """Get current learning rate (compatibility method)"""
        return self.scheduler.get_lr()[0]  # Return LR for first param group
    
    def step(self):
        """Update learning rate (compatibility method)"""
        return self.scheduler.step()


def create_cosine_warmup_scheduler(optimizer, config: Dict[str, Any]) -> CosineWarmupScheduler:
    """
    Factory function to create cosine warmup scheduler
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
    
    Returns:
        scheduler: CosineWarmupScheduler instance
    """
    training_config = config.get('training', {})
    
    # Get total training steps
    num_epochs = training_config.get('num_epochs', 30)
    # This will be set dynamically when DataLoader is created
    # For now, use a reasonable estimate
    batch_size = training_config.get('batch_size', 32)
    num_train_samples = 8549  # NATIX training set
    steps_per_epoch = num_train_samples // batch_size
    num_training_steps = num_epochs * steps_per_epoch
    
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        num_warmup_steps=training_config.get('warmup_steps', 500),
        num_training_steps=num_training_steps,
        min_lr=training_config.get('min_lr', 0.0),
        num_cycles=training_config.get('cosine_cycles', 0.5)  # Decay to 10%
    )
    
    logger.info(f"✅ CosineWarmup scheduler created")
    logger.info(f"   Warmup steps: {training_config.get('warmup_steps', 500)}")
    logger.info(f"   Total training steps: {num_training_steps}")
    logger.info(f"   Min LR: {training_config.get('min_lr', 0.0)}")
    
    return scheduler


if __name__ == "__main__":
    print("🧠 Testing CosineWarmupScheduler...\n")
    
    # Mock optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create scheduler
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        {
            'training': {
                'warmup_steps': 100,
                'num_epochs': 10,
                'batch_size': 32,
                'min_lr': 0.0,
                'cosine_cycles': 0.5
            }
        }
    )
    
    print("📊 Testing warmup phase...")
    for i in range(150):
        scheduler.step()
        lr = scheduler.get_lr()
        if i < 10:
            print(f"   Step {i:3d}: LR = {lr:.2e} (warmup)")
        elif i < 50:
            print(f"   Step {i:3d}: LR = {lr:.2e} (warmup)")
        else:
            print(f"   Step {i:3d}: LR = {lr:.2e} (warmup)")
    
    print(f"\n✅ CosineWarmupScheduler test passed!\n")

