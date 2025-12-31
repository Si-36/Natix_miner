"""
Threshold selection for binary classification

Used in Phase-2 to find optimal confidence threshold that maximizes MCC.

2025 best practices:
- Type hints for torch/numpy arrays
- Vectorized operations (no loops over samples)
- Returns both threshold and best metric
- Supports plotting for debugging
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .metrics import compute_mcc, _to_numpy

# Import matplotlib only when needed (optional dependency)
_plt_module: Optional[object] = None
HAS_MATPLOTLIB = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
    _plt_module = plt
except ImportError:
    HAS_MATPLOTLIB = False
    _plt_module = None


def select_threshold_max_mcc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 2000,  # IMPROVED: Higher resolution finds true optimum (was 100)
) -> Tuple[float, float]:
    """
    Select threshold that maximizes MCC

    Args:
        logits: Model logits (N, 2) or probabilities (N,)
        labels: Ground truth labels (N,)
        n_thresholds: Number of thresholds to sweep

    Returns:
        Tuple of (best_threshold, best_mcc)

    Implementation:
        1. Convert logits to probabilities (sigmoid or softmax)
        2. Sweep thresholds from 0 to 1
        3. Compute MCC for each threshold
        4. Return threshold with max MCC

    Example:
        >>> logits = torch.randn(1000, 2)
        >>> labels = torch.randint(0, 2, (1000,))
        >>> threshold, mcc = select_threshold_max_mcc(logits, labels)
        >>> print(f"Best threshold: {threshold:.3f}, MCC: {mcc:.3f}")
        Best threshold: 0.520, MCC: 0.856
    """
    # Convert to numpy
    logits_np = _to_numpy(logits)
    labels_np = _to_numpy(labels)

    # Get probabilities for positive class
    if len(logits_np.shape) == 2 and logits_np.shape[1] == 2:
        # Two-class logits: apply softmax and take prob of class 1
        probs = torch.softmax(torch.from_numpy(logits_np), dim=1)[:, 1].numpy()
    elif len(logits_np.shape) == 2 and logits_np.shape[1] == 1:
        # Single output: apply sigmoid
        probs = torch.sigmoid(torch.from_numpy(logits_np[:, 0])).numpy()
    elif len(logits_np.shape) == 1:
        # Already probabilities or single output
        if logits_np.max() > 1.0 or logits_np.min() < 0.0:
            # Apply sigmoid if not in [0, 1]
            probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
        else:
            probs = logits_np
    else:
        raise ValueError(f"Unexpected logits shape: {logits_np.shape}")

    # Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_mcc = -1.0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        mcc = compute_mcc(labels_np, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return float(best_threshold), float(best_mcc)


def sweep_thresholds_binary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep thresholds and return MCC curve

    Args:
        logits: Model logits (N, 2) or probabilities (N,)
        labels: Ground truth labels (N,)
        n_thresholds: Number of thresholds to sweep

    Returns:
        Tuple of (thresholds, mcc_scores)

    Use case:
        - Plotting threshold vs MCC curve
        - Analyzing threshold sensitivity
        - Debugging Phase-2 threshold selection

    Example:
        >>> logits = torch.randn(1000, 2)
        >>> labels = torch.randint(0, 2, (1000,))
        >>> thresholds, mccs = sweep_thresholds_binary(logits, labels)
        >>> plt.plot(thresholds, mccs)
        >>> plt.xlabel("Threshold")
        >>> plt.ylabel("MCC")
        >>> plt.savefig("threshold_curve.png")
    """
    # Convert to numpy
    logits_np = _to_numpy(logits)
    labels_np = _to_numpy(labels)

    # Get probabilities
    if len(logits_np.shape) == 2 and logits_np.shape[1] == 2:
        probs = torch.softmax(torch.from_numpy(logits_np), dim=1)[:, 1].numpy()
    elif len(logits_np.shape) == 2 and logits_np.shape[1] == 1:
        probs = torch.sigmoid(torch.from_numpy(logits_np[:, 0])).numpy()
    elif len(logits_np.shape) == 1:
        if logits_np.max() > 1.0 or logits_np.min() < 0.0:
            probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
        else:
            probs = logits_np
    else:
        raise ValueError(f"Unexpected logits shape: {logits_np.shape}")

    # Sweep thresholds
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    mcc_scores = []

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        mcc = compute_mcc(labels, y_pred)
        mcc_scores.append(mcc)

    return thresholds, np.array(mcc_scores)


def plot_threshold_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    n_thresholds: int = 100,
) -> None:
    """
    Plot threshold vs MCC curve

    Args:
        logits: Model logits (N, 2) or probabilities (N,)
        labels: Ground truth labels (N,)
        save_path: Optional path to save figure
        n_thresholds: Number of thresholds to sweep

    Example:
        >>> logits = torch.randn(1000, 2)
        >>> labels = torch.randint(0, 2, (1000,))
        >>> plot_threshold_curve(logits, labels, "threshold_curve.png")

    Raises:
        ImportError: If matplotlib is not installed
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    thresholds, mcc_scores = sweep_thresholds_binary(logits, labels, n_thresholds)

    # Find best threshold
    best_idx = np.argmax(mcc_scores)
    best_threshold = thresholds[best_idx]
    best_mcc = mcc_scores[best_idx]

    # Create plot
    if HAS_MATPLOTLIB and _plt_module is not None:
        _plt_module.figure(figsize=(10, 6))  # type: ignore[attr-defined]
        _plt_module.plot(thresholds, mcc_scores, linewidth=2, label="MCC")  # type: ignore[attr-defined]
        _plt_module.axvline(  # type: ignore[attr-defined]
            best_threshold,
            color="red",
            linestyle="--",
            label=f"Best: {best_threshold:.3f} (MCC={best_mcc:.3f})",
        )
        _plt_module.xlabel("Threshold", fontsize=12)  # type: ignore[attr-defined]
        _plt_module.ylabel("MCC", fontsize=12)  # type: ignore[attr-defined]
        _plt_module.title("Threshold Selection: Maximize MCC", fontsize=14)  # type: ignore[attr-defined]
        _plt_module.legend(fontsize=10)  # type: ignore[attr-defined]
        _plt_module.grid(True, alpha=0.3)  # type: ignore[attr-defined]
        _plt_module.tight_layout()  # type: ignore[attr-defined]

        if save_path:
            _plt_module.savefig(save_path, dpi=150)  # type: ignore[attr-defined]
            print(f"Saved threshold curve to {save_path}")
        else:
            _plt_module.show()  # type: ignore[attr-defined]

        _plt_module.close()  # type: ignore[attr-defined]
