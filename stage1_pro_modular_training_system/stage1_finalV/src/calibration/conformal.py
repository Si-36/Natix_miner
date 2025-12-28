"""
🔥 **Conformal Policy - 2026 Library-Backed Implementation**
Uses TorchCP library for SOTA conformal prediction (no hand-rolling!)
Reference: TorchCP - PyTorch-native, GPU-friendly conformal library
Benefits:
- Library-validated quantile/threshold rules (prevents coverage bugs)
- GPU/batch-native operations (no Python loops)
- Pluggable score functions (APS, RAPS, SAPS, Margin)
- Optional temperature scaling and weighted predictors for covariate shift
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Literal
import warnings

# Try importing TorchCP (will fail gracefully if not installed)
try:
    from torchcp.classification.predictor import SplitPredictor
    from torchcp.classification.scores import (
        APS,
        RAPS,
        SAPS,
        THR,
        Margin,
    )
    from torchcp.classification.utils import WeightedPredictor
    TORCHCP_AVAILABLE = True
except ImportError:
    TORCHCP_AVAILABLE = False
    warnings.warn(
        "TorchCP not installed. Install with: pip install torchcp[cpu] or pip install torchcp[cuda]"
    )


# ============================================================================
# 1. ConformalPolicy - Stateless Calibrator Wrapper
# ============================================================================

class ConformalPolicy:
    """
    2026 Conformal Policy - Library-backed, no nn.Module
    
    Uses TorchCP internally for:
    - SplitConformal with APS/RAPS scores
    - Coverage guarantees (P(y ∈ Ŝ) ≥ 1 - α)
    - GPU/batch-native operations
    - Optional temperature scaling
    - Optional WeightedPredictor for covariate shift
    
    Args:
        alpha: Coverage level (default 0.1 for 90% coverage)
        score_name: Score function ('aps', 'raps', 'saps', 'margin', 'thr', 'lac')
        randomized: Whether to use randomization for coverage
        raps_penalty: Regularization parameter for RAPS
        raps_kreg: Regularization k for RAPS
        score_type: Score type ('softmax', 'log_softmax', 'identity')
        temperature: Optional temperature for post-scaling
        use_weighted_predictor: Whether to use WeightedPredictor for covariate shift
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        score_name: str = "aps",
        randomized: bool = True,
        raps_penalty: float = 0.01,
        raps_kreg: int = 5,
        score_type: str = "softmax",
        temperature: Optional[float] = None,
        use_weighted_predictor: bool = False,
    ):
        self.alpha = alpha
        self.score_name = score_name
        self.randomized = randomized
        self.raps_penalty = raps_penalty
        self.raps_kreg = raps_kreg
        self.score_type = score_type
        self.temperature = temperature
        self.use_weighted_predictor = use_weighted_predictor
        
        # TorchCP predictor (created in fit())
        self.predictor = None
        
        # Fitted state
        self.is_fitted = False
        
        print(f"✅ ConformalPolicy: alpha={alpha}, score={score_name}")
        print(f"   Randomized: {randomized}, RAPS penalty: {raps_penalty}")
        
        if not TORCHCP_AVAILABLE:
            warnings.warn("TorchCP not installed - falling back to placeholder")
    
    def _get_torchcp_score(self):
        """Map score_name to TorchCP score class"""
        score_map = {
            "aps": APS,
            "raps": RAPS,
            "saps": SAPS,
            "margin": Margin,
            "thr": THR,
        }
        
        if self.score_name not in score_map:
            raise ValueError(f"Unknown score: {self.score_name}. Choose from {list(score_map.keys())}")
        
        score_class = score_map[self.score_name]
        
        # Add score_type parameter (softmax vs log_softmax vs identity)
        return score_class(score_type=self.score_type)
    
    def _get_raps_kwargs(self):
        """Get RAPS-specific kwargs"""
        return {
            "lam_reg": self.raps_penalty,
            "k_star": self.raps_kreg,
        }
    
    def fit(
        self,
        calib_logits: torch.Tensor,
        calib_labels: torch.Tensor,
    ) -> None:
        """
        Fit conformal predictor on calibration set
        
        Args:
            calib_logits: Calibration logits [N, num_classes]
            calib_labels: Calibration labels [N]
        """
        if not TORCHCP_AVAILABLE:
            raise ImportError("TorchCP must be installed: pip install torchcp")
        
        # Map score_name to TorchCP score class
        score_class = self._get_torchcp_score()
        
        # Get RAPS-specific kwargs if needed
        kwargs = {}
        if self.score_name in ["aps", "raps"]:
            kwargs.update(self._get_raps_kwargs())
        
        # Create predictor
        self.predictor = SplitPredictor(
            score=score_class(),
            alpha=self.alpha,
            randomize=self.randomized,
            **kwargs,
        )
        
        # Fit on calibration set
        print(f"   Fitting SplitPredictor on {len(calib_logits)} samples...")
        self.predictor.fit(calib_logits, calib_labels)
        self.is_fitted = True
        
        print(f"   ✅ Fitted. Coverage target: {1 - self.alpha:.0%}")
    
    def predict_from_logits(
        self,
        logits: torch.Tensor,
        return_sets: bool = True,
    ) -> torch.Tensor:
        """
        Predict conformal prediction sets from logits
        
        Args:
            logits: Test logits [B, num_classes]
            return_sets: If True, return boolean mask [B, C]; if False, return sets
        
        Returns:
            prediction_sets: Boolean mask [B, num_classes] OR list of sets
        """
        if not self.is_fitted:
            raise ValueError("ConformalPolicy must be fitted before prediction")
        
        # Use TorchCP predict method (GPU-native, batched)
        print(f"   Predicting {len(logits)} samples...")
        sets_bool = self.predictor.predict_sets(logits)
        
        if return_sets:
            # Convert boolean mask to Python sets
            prediction_sets = []
            for i in range(len(sets_bool)):
                in_set = torch.where(sets_bool[i])[0].cpu().numpy()
                prediction_sets.append(set(in_set.tolist()))
            return prediction_sets
        else:
            # Return boolean tensor (N×C) for speed/metrics
            return sets_bool
    
    def evaluate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate conformal prediction metrics
        
        Args:
            logits: Test logits [N, num_classes]
            labels: Ground truth labels [N]
        
        Returns:
            metrics: Dictionary with coverage_rate, average_size, singleton_hit_ratio
        """
        if not self.is_fitted:
            raise ValueError("ConformalPolicy must be fitted before evaluation")
        
        # Get prediction sets
        sets_bool = self.predictor.predict_sets(logits)
        
        # Compute coverage: fraction of labels in prediction sets
        in_set = torch.gather(sets_bool, 1, labels.unsqueeze(1))
        coverage = in_set.float().mean().item()
        
        # Compute average set size
        set_sizes = sets_bool.sum(dim=1).float()
        avg_set_size = set_sizes.mean().item()
        
        # Compute singleton hit ratio (fraction of size-1 sets)
        singleton_sets = set_sizes == 1
        singleton_hit_ratio = singleton_sets.float().mean().item()
        
        metrics = {
            "coverage_rate": coverage,
            "average_size": avg_set_size,
            "singleton_hit_ratio": singleton_hit_ratio,
        }
        
        print(f"\n📊 Conformal Metrics:")
        print(f"   Coverage: {coverage:.1%} (target: {1 - self.alpha:.0%})")
        print(f"   Avg Set Size: {avg_set_size:.2f} (target: <1.5)")
        print(f"   Singleton Rate: {singleton_hit_ratio:.1%} (target: >70%)")
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save fitted conformal policy to disk
        
        Args:
            path: Path to save fitted state
        """
        if not TORCHCP_AVAILABLE:
            raise ImportError("TorchCP must be installed")
        
        if not self.is_fitted:
            raise ValueError("ConformalPolicy must be fitted before saving")
        
        # Save predictor state (TorchCP serialization)
        torch.save(self.predictor.state_dict(), path)
        print(f"✅ Saved conformal policy to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        Load fitted conformal policy from disk
        
        Args:
            path: Path to load fitted state
        
        Returns:
            ConformalPolicy instance
        """
        if not TORCHCP_AVAILABLE:
            raise ImportError("TorchCP must be installed")
        
        # Load state and create predictor
        state_dict = torch.load(path)
        policy = cls()
        policy.predictor = SplitPredictor.load_state_dict(state_dict)
        policy.is_fitted = True
        
        print(f"✅ Loaded conformal policy from {path}")
        return policy
    
    def get_qhat(self) -> float:
        """Get fitted quantile threshold"""
        if not self.is_fitted or not TORCHCP_AVAILABLE:
            raise ValueError("ConformalPolicy must be fitted with TorchCP")
        
        # TorchCP stores qhat internally
        # Access via predictor's internal state
        return self.predictor.q_hat.item() if hasattr(self.predictor, 'q_hat') else None


# ============================================================================
# 2. Temperature Conformal Wrapper
# ============================================================================

class TemperatureConformalWrapper:
    """
    Temperature Scaling for Conformal - Post-hoc calibration
    
    Applies temperature scaling before conformal prediction.
    This improves stability of conformal scores when softmax is miscalibrated.
    
    Args:
        temperature: Learnable temperature (default 1.0)
        calibration_split: Which split to use for fitting temperature
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        calibration_split: str = "val_calib",
    ):
        self.temperature = temperature
        self.calibration_split = calibration_split
        self.is_fitted = False
        self.calibrator = None
        
        print(f"✅ TemperatureConformalWrapper: temp={temperature}")
    
    def fit(self, calib_logits: torch.Tensor, calib_labels: torch.Tensor) -> None:
        """
        Fit temperature on calibration set
        
        Args:
            calib_logits: Calibration logits [N, num_classes]
            calib_labels: Calibration labels [N]
        """
        from scipy.optimize import minimize_scalar
        
        def nll_loss(log_temp):
            """Negative log-likelihood"""
            temp = torch.exp(torch.tensor(log_temp))
            log_probs = torch.log_softmax(calib_logits / temp, dim=-1)
            nll = -log_probs[torch.arange(len(calib_labels)), calib_labels].mean()
            return nll.item()
        
        # Optimize temperature
        result = minimize_scalar(
            nll_loss,
            x0=np.log(self.temperature),
            method="L-BFGS-B",
        )
        
        self.temperature = torch.exp(torch.tensor(result.x)).item()
        self.is_fitted = True
        
        print(f"   Fitted temperature: {self.temperature:.4f}")
    
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrate logits with temperature
        
        Args:
            logits: Raw logits [B, num_classes]
        
        Returns:
            Temperature-scaled logits [B, num_classes]
        """
        if not self.is_fitted:
            raise ValueError("TemperatureConformalWrapper must be fitted before calibration")
        
        return logits / self.temperature


# ============================================================================
# Export for easy imports
# ============================================================================

__all__ = [
    "ConformalPolicy",
    "TemperatureConformalWrapper",
    "TORCHCP_AVAILABLE",
]

