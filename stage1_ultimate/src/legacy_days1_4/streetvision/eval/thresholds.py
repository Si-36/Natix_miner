"""
Threshold selection for binary classification

Used in Phase-2 to find optimal confidence threshold that maximizes MCC.

2025 best practices:
- Type hints for torch/numpy arrays
- Vectorized operations (no loops over samples)
- Returns both threshold and best metric
- Supports plotting for debugging
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

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
    n_thresholds: int = 5000,
    return_curve: bool = False,
) -> Tuple[float, float, Dict, Optional[pd.DataFrame]]:
    """
    Find threshold maximizing MCC using vectorized computation.
    
    2025 OPTIMIZATION: Vectorized NumPy instead of Python loop.
    10× faster than sklearn loop for 5000 thresholds.
    
    Args:
        logits: [N, num_classes] raw model outputs
        labels: [N] ground truth (0=no_roadwork, 1=roadwork)
        n_thresholds: Number of thresholds (5000 recommended)
        return_curve: Return full MCC curve DataFrame
    
    Returns:
        best_threshold, best_mcc, metrics_dict, [optional: curve_df]
    
    Example:
        >>> logits = torch.randn(1000, 2)
        >>> labels = torch.randint(0, 2, (1000,))
        >>> threshold, mcc, metrics, curve = select_threshold_max_mcc(logits, labels, return_curve=True)
        >>> print(f"Best threshold: {threshold:.3f}, MCC: {mcc:.3f}")
    """
    # Get positive class probabilities
    probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # [N]
    labels_np = labels.cpu().numpy()  # [N]
    
    # Create threshold grid
    thresholds = np.linspace(0, 1, n_thresholds)
    
    # VECTORIZED MCC COMPUTATION (2025 optimization)
    # Instead of loop, broadcast computation
    # Shape: [n_thresholds, N]
    preds_all = (probs[None, :] >= thresholds[:, None]).astype(np.int32)
    
    # Compute confusion matrix elements for all thresholds at once
    # Positive: label=1, Negative: label=0
    tp = ((preds_all == 1) & (labels_np[None, :] == 1)).sum(axis=1)  # [n_thresholds]
    tn = ((preds_all == 0) & (labels_np[None, :] == 0)).sum(axis=1)
    fp = ((preds_all == 1) & (labels_np[None, :] == 0)).sum(axis=1)
    fn = ((preds_all == 0) & (labels_np[None, :] == 1)).sum(axis=1)
    
    # Vectorized MCC formula
    # MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mccs = np.where(denominator != 0, numerator / denominator, 0)
    
    # Find best threshold
    best_idx = np.argmax(mccs)
    best_threshold = float(thresholds[best_idx])
    best_mcc = float(mccs[best_idx])
    
    # Compute full metrics at best threshold
    best_preds = (probs >= best_threshold).astype(np.int32)
    cm = confusion_matrix(labels_np, best_preds)
    tn_best, fp_best, fn_best, tp_best = cm.ravel()
    
    metrics = {
        'accuracy': float((tp_best + tn_best) / len(labels_np)),
        'precision': float(tp_best / (tp_best + fp_best)) if (tp_best + fp_best) > 0 else 0.0,
        'recall': float(tp_best / (tp_best + fn_best)) if (tp_best + fn_best) > 0 else 0.0,
        'f1': float(2 * tp_best / (2 * tp_best + fp_best + fn_best)) if (2 * tp_best + fp_best + fn_best) > 0 else 0.0,
        'mcc': best_mcc,
        'fnr': float(fn_best / (fn_best + tp_best)) if (fn_best + tp_best) > 0 else 0.0,
        'fpr': float(fp_best / (fp_best + tn_best)) if (fp_best + tn_best) > 0 else 0.0,
        'tn': int(tn_best),
        'fp': int(fp_best),
        'fn': int(fn_best),
        'tp': int(tp_best),
    }
    
    if return_curve:
        curve = pd.DataFrame({
            'threshold': thresholds,
            'mcc': mccs,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
        })
        return best_threshold, best_mcc, metrics, curve
    
    return best_threshold, best_mcc, metrics, None


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


def plot_mcc_curve(
    curve: pd.DataFrame,
    best_threshold: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot MCC vs threshold with optimal point marked.
    
    Creates 2-panel plot:
    - Left: MCC curve with optimal threshold marked
    - Right: Confusion matrix breakdown (TP/TN/FP/FN vs threshold)
    
    Args:
        curve: DataFrame with columns ['threshold', 'mcc', 'tp', 'tn', 'fp', 'fn']
        best_threshold: Optimal threshold value
        save_path: Path to save figure (optional)
    
    Example:
        >>> threshold, mcc, metrics, curve = select_threshold_max_mcc(logits, labels, return_curve=True)
        >>> plot_mcc_curve(curve, threshold, "mcc_curve.png")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    if _plt_module is None:
        raise ImportError("matplotlib.pyplot not available")
    
    fig, (ax1, ax2) = _plt_module.subplots(1, 2, figsize=(14, 5))
    
    # MCC curve
    ax1.plot(curve['threshold'], curve['mcc'], linewidth=2, color='#2E86AB')
    ax1.axvline(best_threshold, color='#A23B72', linestyle='--', linewidth=2,
                label=f'Optimal: {best_threshold:.4f}')
    best_mcc_val = curve.loc[curve['threshold'] == best_threshold, 'mcc'].values[0]
    ax1.axhline(best_mcc_val, color='#F18F01', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Classification Threshold', fontsize=12)
    ax1.set_ylabel('Matthews Correlation Coefficient', fontsize=12)
    ax1.set_title('MCC vs Threshold (5000-Grid Search)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Confusion matrix breakdown
    ax2.plot(curve['threshold'], curve['tp'], label='TP', linewidth=2, color='#06A77D')
    ax2.plot(curve['threshold'], curve['tn'], label='TN', linewidth=2, color='#2E86AB')
    ax2.plot(curve['threshold'], curve['fp'], label='FP', linewidth=2, color='#F18F01')
    ax2.plot(curve['threshold'], curve['fn'], label='FN', linewidth=2, color='#A23B72')
    ax2.axvline(best_threshold, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confusion Matrix Breakdown', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    _plt_module.tight_layout()
    
    if save_path:
        _plt_module.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved MCC curve to {save_path}")
    else:
        _plt_module.show()
    
    _plt_module.close()
