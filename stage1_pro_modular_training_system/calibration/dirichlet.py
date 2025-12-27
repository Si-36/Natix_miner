"""
Dirichlet calibration for Phase 6+

DirichletCalibrator (matrix scaling on logits) with ODIR regularization.
Fit on val_calib only. Save/load support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DirichletCalibrator(nn.Module):
    """
    Dirichlet calibrator (matrix scaling) - PHASE 6 ONLY.
    
    Architecture: Linear(num_classes, num_classes, bias=True)
    Initialize to identity (safe start)
    Forward: log_probs = log_softmax(logits) → cal_logits = linear(log_probs)
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_classes, num_classes, bias=True)
        # Initialize to identity
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: log_probs = log_softmax(logits), cal_logits = linear(log_probs)
        
        Args:
            logits: Input logits [B, num_classes]
        
        Returns:
            Calibrated logits [B, num_classes]
        """
        log_probs = F.log_softmax(logits, dim=1)
        cal_logits = self.linear(log_probs)
        return cal_logits


class ODIRRegularizer:
    """
    ODIR Regularizer: Off-Diagonal + Intercept regularization - PHASE 6 ONLY.
    
    Penalty: lambda_odir * (||W - diag(W)||_F^2 + ||b||_2^2)
    Prevents overfitting of calibration mapping.
    """
    
    def __init__(self, lambda_odir: float = 0.01):
        self.lambda_odir = lambda_odir
    
    def compute_penalty(self, calibrator: DirichletCalibrator) -> torch.Tensor:
        """
        Compute ODIR penalty.
        
        Args:
            calibrator: DirichletCalibrator instance
        
        Returns:
            Penalty tensor
        """
        W = calibrator.linear.weight
        b = calibrator.linear.bias
        
        # Off-diagonal penalty
        diag_W = torch.diag(torch.diag(W))
        off_diag_penalty = torch.norm(W - diag_W, p='fro') ** 2
        
        # Intercept penalty
        intercept_penalty = torch.norm(b, p=2) ** 2
        
        return self.lambda_odir * (off_diag_penalty + intercept_penalty)
