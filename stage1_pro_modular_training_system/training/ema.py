"""
EMA (Exponential Moving Average) preserving EXACT implementation from train_stage1_head.py

2025 SOTA for vision models - stabilizes training and improves generalization.
"""

import torch


class EMA:
    """
    Exponential Moving Average (2025 SOTA for vision models)
    
    Preserves EXACT implementation from train_stage1_head.py:
    - register(): Initialize shadow parameters
    - update(): Update shadow with decay
    - apply_shadow(): Replace params with shadow
    - restore(): Restore original params
    - decay: 0.9999 (preserved)
    """
    
    def __init__(self, model, decay=0.9999):
        """
        Initialize EMA.
        
        Args:
            model: Model to track EMA for
            decay: EMA decay factor (default 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        """
        Preserve EXACT logic from train_stage1_head.py:
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow[name] = param.data.clone()
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """
        Preserve EXACT logic from train_stage1_head.py:
        new_average = (1.0 - decay) * param.data + decay * shadow[name]
        shadow[name] = new_average.clone()
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """
        Preserve EXACT logic from train_stage1_head.py:
        backup params, replace with shadow
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """
        Preserve EXACT logic from train_stage1_head.py:
        restore original params from backup
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
