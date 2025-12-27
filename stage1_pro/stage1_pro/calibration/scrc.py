import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import json


class SCRCCalibrator:
    """
    SCRC-I calibrator: Selective Classification Risk Conformal Inference.

    Phase 2: Quantile-based calibration for satisfying FNR constraints.
    Uses quantiles of confidence scores to find optimal thresholds.
    """

    def __init__(self, alpha: float = 0.02, n_bins: int = 10):
        """
        Args:
            alpha: Target risk level (FNR constraint, default 2%)
            n_bins: Number of bins for quantile estimation
        """
        self.alpha = alpha
        self.n_bins = n_bins
        self.quantiles = None
        self.num_classes = None
        self.fitted = False

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit SCRC-I quantiles to calibration data.

        Args:
            logits: (N, num_classes) uncalibrated logits
            labels: (N,) ground truth labels
        """
        self.num_classes = logits.shape[1]

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1).numpy()
        labels_np = labels.numpy()

        # Compute max probability per sample
        max_probs = np.max(probs, axis=1)

        # Compute confidence calibration curves per class
        self.quantiles = np.zeros(self.n_bins)

        for bin_idx in range(self.n_bins):
            # Bin quantile level
            quantile_level = (bin_idx + 1) / self.n_bins

            # Compute quantile threshold
            self.quantiles[bin_idx] = np.quantile(max_probs, quantile_level)

        self.fitted = True

    def predict(
        self, logits: torch.Tensor, target_fnr: float = 0.02
    ) -> Tuple[np.ndarray, float]:
        """
        Predict optimal threshold for target FNR.

        Args:
            logits: (N, num_classes) uncalibrated logits
            target_fnr: Target false negative rate

        Returns:
            calibrated_probs: (N, num_classes) calibrated probabilities
            optimal_threshold: Optimal threshold for target FNR
        """
        if not self.fitted:
            raise RuntimeError("SCRCCalibrator must be fit before predict")

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)

        max_probs_np = max_probs.numpy()
        labels_np = preds.numpy()

        # Sweep thresholds to find one satisfying FNR constraint
        thresholds = np.linspace(0.0, 1.0, 100)
        best_threshold = 0.5
        best_coverage = 0.0
        best_fnr = 1.0

        for threshold in thresholds:
            exit_mask = max_probs_np >= threshold

            if exit_mask.sum() == 0:
                continue

            exit_labels = labels_np[exit_mask]
            exit_preds = preds[exit_mask].numpy()

            # Compute FNR on exits
            if (exit_labels == 1).sum() > 0:
                fnr = ((exit_labels == 1) & (exit_preds == 0)).sum() / (
                    exit_labels == 1
                ).sum()
            else:
                fnr = 1.0

            # Check if satisfies constraint
            if fnr <= target_fnr:
                # Among valid, maximize coverage
                coverage = exit_mask.mean()
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_fnr = fnr
                    best_threshold = threshold
            elif best_fnr > target_fnr:
                # Update best even if constraint not met
                best_fnr = fnr
                best_threshold = threshold

        return probs.numpy(), best_threshold

    def save(self, path: str):
        """Save calibrator to file."""
        state = {
            "alpha": self.alpha,
            "n_bins": self.n_bins,
            "quantiles": self.quantiles.tolist()
            if self.quantiles is not None
            else None,
            "num_classes": self.num_classes,
            "fitted": self.fitted,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load calibrator from file."""
        with open(path, "r") as f:
            state = json.load(f)

        calibrator = cls(alpha=state["alpha"], n_bins=state["n_bins"])
        calibrator.quantiles = (
            np.array(state["quantiles"]) if state["quantiles"] is not None else None
        )
        calibrator.num_classes = state["num_classes"]
        calibrator.fitted = state["fitted"]

        return calibrator
