"""
Loss functions for Stage-1 Pro Modular Training System

Phase 1: CrossEntropyLoss with class weights + label smoothing (preserved from baseline)
Phase 3+: SelectiveLoss + AuxiliaryLoss
Phase 6+: ConformalRiskLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Preserve exact class weight computation from train_stage1_head.py:
    class_weights = total_samples / (num_classes * class_counts)
    
    Args:
        labels: Tensor of class labels
        num_classes: Number of classes
    
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(labels.cpu().numpy())
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(class_weights)


class SelectiveLoss(nn.Module):
    """
    Selective loss for Phase 3+.
    
    CRITICAL: Control FNR ≤ target_fnr_exit, maximize coverage implicitly.
    NO target_coverage parameter - coverage is maximized by minimizing selective risk
    subject to FNR constraint.
    """
    
    def __init__(self, target_fnr_exit: float = 0.02, gate_threshold: float = 0.5):
        """
        Args:
            target_fnr_exit: Target FNR on exited samples (e.g., 0.02 for 2%)
            gate_threshold: Threshold for gate acceptance
        """
        super().__init__()
        self.target_fnr_exit = target_fnr_exit
        self.gate_threshold = gate_threshold
    
    def forward(
        self,
        logits: torch.Tensor,
        gate_logit: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute selective risk (error on accepted samples).
        
        Objective: Minimize selective risk subject to FNR ≤ target_fnr_exit.
        Coverage maximized implicitly by minimizing selective risk.
        
        Args:
            logits: Classification logits [B, 2]
            gate_logit: Gate logits [B]
            labels: Ground truth labels [B]
        
        Returns:
            Selective risk loss
        """
        gate_prob = torch.sigmoid(gate_logit)
        accepted_mask = gate_prob >= self.gate_threshold
        
        if accepted_mask.sum() == 0:
            # No samples accepted - return high loss
            return torch.tensor(1.0, device=logits.device, requires_grad=True)
        
        # Compute error on accepted samples
        predictions = torch.argmax(logits[accepted_mask], dim=1)
        errors = (predictions != labels[accepted_mask]).float()
        selective_risk = errors.mean()
        
        # Compute FNR on exited samples (missed positives)
        positive_mask = labels == 1
        exited_positives = positive_mask & accepted_mask
        if exited_positives.sum() > 0:
            fnr = (predictions[exited_positives] == 0).float().mean()
            # Penalize if FNR exceeds target
            fnr_penalty = F.relu(fnr - self.target_fnr_exit)
        else:
            fnr_penalty = torch.tensor(0.0, device=logits.device)
        
        return selective_risk + fnr_penalty


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for Phase 3+.
    
    Standard CrossEntropyLoss on aux_logits with full coverage (all samples).
    Prevents collapse during selective training.
    """
    
    def __init__(self, weight: float = 0.5):
        """
        Args:
            weight: Weight for auxiliary loss
        """
        super().__init__()
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        aux_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss on all samples.
        
        Args:
            aux_logits: Auxiliary classification logits [B, 2]
            labels: Ground truth labels [B]
        
        Returns:
            Weighted auxiliary loss
        """
        return self.weight * self.criterion(aux_logits, labels)


class ConformalRiskLoss(nn.Module):
    """
    End-to-end conformal risk loss for Phase 6.
    
    Implements NeurIPS 2025 "End-to-End Optimization of Conformal Risk Control".
    Batch splitting (pseudo-calib/pseudo-pred), FNR≤2% control, gradient through CRC.
    """
    
    def __init__(self, target_fnr: float = 0.02, batch_split_ratio: float = 0.5):
        """
        Args:
            target_fnr: Target FNR on exited samples
            batch_split_ratio: Ratio for calib/pred split
        """
        super().__init__()
        self.target_fnr = target_fnr
        self.batch_split_ratio = batch_split_ratio
    
    def forward(
        self,
        logits: torch.Tensor,
        gate_logit: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conformal risk loss with batch splitting.
        
        Args:
            logits: Classification logits [B, 2]
            gate_logit: Gate logits [B]
            labels: Ground truth labels [B]
        
        Returns:
            Conformal risk loss
        """
        # TODO: Implement conformal risk loss (Phase 6)
        # Split batch: pseudo-calib (50%) + pseudo-pred (50%)
        # Compute conformal threshold on calib set
        # Differentiate through CRC objective on pred set
        raise NotImplementedError("ConformalRiskLoss implementation - Phase 6 only")
