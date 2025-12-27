"""
Stage1Head module preserving exact architecture from train_stage1_head.py

Phase 1: Single-head (matches baseline exactly)
Phase 3+: Extends to 3-head (cls + gate + aux)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Stage1Head(nn.Module):
    """
    Stage-1 classifier head preserving exact architecture from train_stage1_head.py.
    
    Phase 1: Single-head architecture
        Linear(hidden_size, 768) → ReLU → Dropout → Linear(768, 2)
    
    Phase 3+: 3-head architecture
        Shared trunk: Linear(hidden_size, 768) → ReLU → Dropout
        cls_head: Linear(768, 2) - Classification logits
        gate_head: Linear(768, 1) - Selection score (sigmoid → exit prob)
        aux_head: Linear(768, 2) - Auxiliary classifier (training only)
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.3,
        phase: int = 1
    ):
        """
        Initialize Stage1Head.
        
        Args:
            hidden_size: Input feature dimension (e.g., 1536 for DINOv3-ViT-H/16+)
            dropout: Dropout probability
            phase: Phase number (1 = single-head, 3+ = 3-head)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.phase = phase
        
        # Shared trunk (preserved from baseline)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Phase 1: Single-head (matches baseline exactly)
        if phase == 1:
            self.cls_head = nn.Linear(768, 2)
            self.gate_head = None
            self.aux_head = None
        else:
            # Phase 3+: 3-head architecture
            self.cls_head = nn.Linear(768, 2)
            self.gate_head = nn.Linear(768, 1)
            self.aux_head = nn.Linear(768, 2)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass preserving exact logic.
        
        Phase 1: Returns logits only
        Phase 3+: Returns (logits, gate_logit, aux_logits)
        
        Args:
            features: Input features [B, hidden_size]
        
        Returns:
            Phase 1: logits [B, 2]
            Phase 3+: (logits [B, 2], gate_logit [B], aux_logits [B, 2])
        """
        trunk_out = self.trunk(features)
        
        if self.phase == 1:
            logits = self.cls_head(trunk_out)
            return logits
        else:
            logits = self.cls_head(trunk_out)
            gate_logit = self.gate_head(trunk_out).squeeze(-1)  # [B]
            aux_logits = self.aux_head(trunk_out)
            return logits, gate_logit, aux_logits
    
    def compile(self, mode: str = "default"):
        """
        Preserve exact compilation logic from train_stage1_head.py:
        torch.compile(model, mode='default')
        
        Args:
            mode: Compilation mode
        
        Returns:
            Compiled model
        """
        return torch.compile(self, mode=mode)
