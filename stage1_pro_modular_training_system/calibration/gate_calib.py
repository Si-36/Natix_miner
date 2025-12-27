"""
Gate calibration for Phase 3+

Platt scaling (logistic regression) and Isotonic regression for gate calibration.
Fits on val_calib gate logits + correctness labels.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Tuple
import json
from pathlib import Path


class PlattCalibrator:
    """
    Platt scaling (logistic regression) for gate calibration.
    
    Fits sigmoid(scale * gate_logit + bias) to correctness labels.
    """
    
    def __init__(self):
        self.scale = None
        self.bias = None
        self.fitted = False
    
    def fit(self, gate_logits: np.ndarray, correctness: np.ndarray):
        """
        Fit Platt calibrator.
        
        Args:
            gate_logits: Gate logits [N]
            correctness: Correctness labels [N] (1 if correct, 0 if wrong)
        """
        # Use sklearn LogisticRegression for stable fitting
        lr = LogisticRegression()
        lr.fit(gate_logits.reshape(-1, 1), correctness)
        
        # Extract scale and bias
        self.scale = lr.coef_[0][0]
        self.bias = lr.intercept_[0]
        self.fitted = True
    
    def predict(self, gate_logits: np.ndarray) -> np.ndarray:
        """
        Predict calibrated gate probabilities.
        
        Args:
            gate_logits: Gate logits [N]
        
        Returns:
            Calibrated gate probabilities [N]
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        
        return 1.0 / (1.0 + np.exp(-(self.scale * gate_logits + self.bias)))
    
    def predict_torch(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Predict calibrated gate probabilities (PyTorch version).
        
        Args:
            gate_logits: Gate logits [N]
        
        Returns:
            Calibrated gate probabilities [N]
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        
        # Convert scale and bias to torch tensors on same device as gate_logits
        scale = torch.tensor(self.scale, dtype=gate_logits.dtype, device=gate_logits.device)
        bias = torch.tensor(self.bias, dtype=gate_logits.dtype, device=gate_logits.device)
        
        return torch.sigmoid(scale * gate_logits + bias)
    
    def save(self, path: str):
        """Save calibrator parameters"""
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        
        torch.save({
            'scale': self.scale,
            'bias': self.bias,
            'type': 'platt'
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load calibrator parameters"""
        data = torch.load(path)
        calibrator = cls()
        calibrator.scale = data['scale']
        calibrator.bias = data['bias']
        calibrator.fitted = True
        return calibrator


class IsotonicCalibrator:
    """
    Isotonic regression for gate calibration (non-parametric fallback).
    """
    
    def __init__(self):
        self.calibrator = None
        self.fitted = False
    
    def fit(self, gate_logits: np.ndarray, correctness: np.ndarray):
        """
        Fit Isotonic calibrator.
        
        Args:
            gate_logits: Gate logits [N]
            correctness: Correctness labels [N] (1 if correct, 0 if wrong)
        """
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(gate_logits, correctness)
        self.fitted = True
    
    def predict(self, gate_logits: np.ndarray) -> np.ndarray:
        """
        Predict calibrated gate probabilities.
        
        Args:
            gate_logits: Gate logits [N]
        
        Returns:
            Calibrated gate probabilities [N]
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        
        return self.calibrator.predict(gate_logits)
    
    def save(self, path: str):
        """Save calibrator (requires pickle)"""
        import pickle
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        
        with open(path, 'wb') as f:
            pickle.dump(self.calibrator, f)
    
    @classmethod
    def load(cls, path: str):
        """Load calibrator"""
        import pickle
        calibrator = cls()
        with open(path, 'rb') as f:
            calibrator.calibrator = pickle.load(f)
        calibrator.fitted = True
        return calibrator


def fit_gate_calibrator(
    gate_logits: torch.Tensor,
    correctness: torch.Tensor,
    method: str = "platt"
) -> Tuple[PlattCalibrator, dict]:
    """
    Fit gate calibrator and compute metrics.
    
    Args:
        gate_logits: Gate logits [N]
        correctness: Correctness labels [N] (1 if correct, 0 if wrong)
        method: "platt" or "isotonic"
    
    Returns:
        Tuple of (calibrator, metrics_dict)
    """
    gate_logits_np = gate_logits.numpy() if isinstance(gate_logits, torch.Tensor) else gate_logits
    correctness_np = correctness.numpy() if isinstance(correctness, torch.Tensor) else correctness
    
    if method == "platt":
        calibrator = PlattCalibrator()
    elif method == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    calibrator.fit(gate_logits_np, correctness_np)
    
    # Compute calibration metrics
    calibrated_probs = calibrator.predict(gate_logits_np)
    
    # ECE for gate calibration
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (calibrated_probs > bin_lower) & (calibrated_probs <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            avg_confidence = calibrated_probs[in_bin].mean()
            avg_accuracy = correctness_np[in_bin].mean()
            ece += (bin_size / len(correctness_np)) * abs(avg_confidence - avg_accuracy)
    
    metrics = {
        'ece': float(ece),
        'method': method
    }
    
    if method == "platt":
        metrics['scale'] = float(calibrator.scale)
        metrics['bias'] = float(calibrator.bias)
    
    return calibrator, metrics
