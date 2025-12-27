"""
Bootstrap utilities for confidence interval computation.

Simple bootstrap resampling for uncertainty estimation.
"""

import numpy as np
from typing import Tuple, Optional


def bootstrap_resample(n: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Bootstrap resample: sample with replacement.
    
    Args:
        n: Sample size
        random_state: Random seed
    
    Returns:
        Array of resampled indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    return np.random.choice(n, size=n, replace=True)


def compute_confidence_intervals(
    samples: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals using percentile method.
    
    Args:
        samples: Bootstrap samples [n_bootstrap, ...]
        confidence: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    alpha = 1 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bound = np.percentile(samples, lower_percentile, axis=0)
    upper_bound = np.percentile(samples, upper_percentile, axis=0)
    
    return lower_bound, upper_bound
