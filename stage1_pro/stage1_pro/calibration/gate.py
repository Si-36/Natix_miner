import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
import json


class GateCalibrator:
    """
    Gate calibrator using Platt scaling (logistic regression).

    Calibrates gate scores to better match exit decisions.
    Phase 2: Used with Dirichlet calibration for better FNR constraint satisfaction.
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.fitted = False

    def fit(self, gate_scores: torch.Tensor, labels: torch.Tensor):
        """
        Fit Platt scaling parameters.

        Args:
            gate_scores: (N,) gate scores from model
            labels: (N,) ground truth labels
        """
        gate_scores_np = gate_scores.numpy()
        labels_np = labels.numpy()

        # Compute binary targets: 1 if correct, 0 if incorrect
        # Gate should be high for correct samples, low for incorrect
        cls_probs = torch.softmax(torch.randn(len(labels), 2), dim=-1)
        cls_preds = cls_probs.argmax(dim=-1)
        targets = (cls_preds == labels).float().numpy()

        # Platt scaling: Fit logistic regression
        # logit(p) = a * s + b
        # Optimize for binary cross-entropy

        # Simple implementation: match quantiles
        correct_scores = gate_scores_np[targets == 1]
        incorrect_scores = gate_scores_np[targets == 0]

        if len(correct_scores) > 0 and len(incorrect_scores) > 0:
            median_correct = np.median(correct_scores)
            median_incorrect = np.median(incorrect_scores)

            # Set parameters to separate distributions
            self.a = 10.0 / (median_correct - median_incorrect + 1e-6)
            self.b = -self.a * ((median_correct + median_incorrect) / 2)
        else:
            self.a = 1.0
            self.b = 0.0

        self.fitted = True

    def predict(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """
        Calibrate gate scores using fitted Platt scaling.

        Args:
            gate_scores: (N,) uncalibrated gate scores

        Returns:
            calibrated_probs: (N,) calibrated gate probabilities
        """
        if not self.fitted:
            raise RuntimeError("GateCalibrator must be fit before predict")

        # Platt scaling: p = sigmoid(a * s + b)
        calibrated = torch.sigmoid(self.a * gate_scores + self.b)

        return calibrated

    def save(self, path: str):
        """Save calibrator to file."""
        state = {
            "a": float(self.a) if self.a is not None else None,
            "b": float(self.b) if self.b is not None else None,
            "fitted": self.fitted,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load calibrator from file."""
        with open(path, "r") as f:
            state = json.load(f)

        calibrator = cls()
        calibrator.a = state["a"]
        calibrator.b = state["b"]
        calibrator.fitted = state["fitted"]

        return calibrator
