import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class DirichletCalibrator:
    """
    Dirichlet calibration for Phase 2.

    Calibrates confidence scores using Dirichlet distribution fitting.
    Matches SCRC-I approach but uses Dirichlet instead of quantiles.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.num_classes = None
        self.alphas = None

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit Dirichlet parameters to calibration data.

        Args:
            logits: (N, num_classes) uncalibrated logits
            labels: (N,) ground truth labels
        """
        self.num_classes = logits.shape[1]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1).numpy()
        labels_np = labels.numpy()

        # Fit Dirichlet parameters per class
        self.alphas = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            # Get samples from class c
            mask = labels_np == c
            if mask.sum() == 0:
                self.alphas[c] = np.ones(self.num_classes) * self.alpha
                continue

            class_probs = probs[mask]

            # Method of moments estimation
            mean_probs = class_probs.mean(axis=0)

            # Estimate alpha from mean: alpha = mean * (sum(mean) - 1) / var * (sum(mean) - 1)
            # Simplified: alpha = mean * concentration
            concentration = self.alpha
            self.alphas[c] = mean_probs * concentration

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrate logits using fitted Dirichlet parameters.

        Args:
            logits: (N, num_classes) uncalibrated logits

        Returns:
            calibrated_probs: (N, num_classes) calibrated probabilities
        """
        if self.alphas is None:
            raise RuntimeError("DirichletCalibrator must be fit before predict")

        # Get predicted class
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        # Apply class-specific calibration
        calibrated_probs = torch.zeros_like(probs)

        for c in range(self.num_classes):
            mask = preds == c

            # Use predicted class's alphas
            alphas_c = torch.from_numpy(self.alphas[c]).float()

            # Dirichlet calibration: p' = p * alpha / sum(p * alpha)
            if mask.any():
                calib_c = probs[mask] * alphas_c.unsqueeze(0)
                calib_c = calib_c / calib_c.sum(dim=-1, keepdim=True)
                calibrated_probs[mask] = calib_c
            else:
                calibrated_probs[mask] = probs[mask]

        return calibrated_probs

    def save(self, path: str):
        """Save calibrator to file."""
        import json

        state = {
            "alpha": self.alpha,
            "num_classes": self.num_classes,
            "alphas": self.alphas.tolist() if self.alphas is not None else None,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load calibrator from file."""
        import json

        with open(path, "r") as f:
            state = json.load(f)

        calibrator = cls(alpha=state["alpha"])
        calibrator.num_classes = state["num_classes"]
        calibrator.alphas = (
            np.array(state["alphas"]) if state["alphas"] is not None else None
        )

        return calibrator
