"""
Binary Classification Head
2026 implementation

Input: 512-dim fused features
Output: 2 classes (roadwork vs no-roadwork)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BinaryClassifier(nn.Module):
    """
    Binary classification head
    
    Input: [B, 512]
    Output: [B, 2] logits
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        logger.info(f"BinaryClassifier: {input_dim} → {num_classes}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
