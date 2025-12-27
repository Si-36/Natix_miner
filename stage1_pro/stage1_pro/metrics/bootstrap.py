import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class BootstrapConfig:
    """Bootstrap ECE configuration."""

    n_bootstrap: int = 100
    n_bins: int = 10
    alpha: float = 0.05  # Confidence interval level
    random_seed: int = 42


class BootstrapECE:
    """
    Bootstrap Expected Calibration Error with confidence intervals.

    Modern 2025 implementation using efficient bootstrap resampling.
    Reference: https://arxiv.org/abs/2010.08976
    """

    def __init__(self, config: Optional[BootstrapConfig] = None):
        self.config = config or BootstrapConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def _compute_ece_single(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = None
    ) -> float:
        """Compute ECE for a single sample."""
        if n_bins is None:
            n_bins = self.config.n_bins

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.sum()

            if bin_size > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += (bin_size / len(labels)) * abs(avg_confidence - avg_accuracy)

        return ece

    def compute(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Compute bootstrap ECE with confidence intervals.

        Args:
            probs: (N, num_classes) probabilities
            labels: (N,) ground truth labels

        Returns:
            Dictionary with ECE statistics
        """
        n_samples = len(probs)
        ece_values = []

        # Bootstrap samples
        for _ in range(self.config.n_bootstrap):
            # Resample with replacement
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            boot_probs = probs[indices]
            boot_labels = labels[indices]

            # Compute ECE
            ece = self._compute_ece_single(boot_probs, boot_labels)
            ece_values.append(ece)

        ece_values = np.array(ece_values)

        # Compute statistics
        results = {
            "ece_mean": float(ece_values.mean()),
            "ece_std": float(ece_values.std()),
            "ece_median": float(np.median(ece_values)),
            "ece_min": float(ece_values.min()),
            "ece_max": float(ece_values.max()),
        }

        # Confidence intervals
        lower = (self.config.alpha / 2) * 100
        upper = (1 - self.config.alpha / 2) * 100
        results[f"ci_{self.config.alpha}"] = {
            "lower": float(np.percentile(ece_values, lower)),
            "upper": float(np.percentile(ece_values, upper)),
        }

        # Original ECE
        results["ece_original"] = self._compute_ece_single(probs, labels)

        return results

    def compute_per_class(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Compute ECE per class with bootstrap."""
        n_classes = probs.shape[1]
        per_class_results = {}

        for class_idx in range(n_classes):
            # Filter to this class
            mask = labels == class_idx
            if mask.sum() == 0:
                continue

            class_probs = probs[mask]
            class_labels = labels[mask]

            # Compute bootstrap ECE for this class
            ece_stats = self.compute(class_probs, class_labels)
            per_class_results[f"class_{class_idx}"] = ece_stats

        return per_class_results
