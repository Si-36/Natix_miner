"""
Gate Head Module - Phase 3: Gate Head Architecture (Dec 2025 Best Practice) - SELECTIVENET-STYLE OBJECTIVE

Implements SelectiveNet-style 3-head design for selective classification:
- Head 1: Classifier (softmax, provides predictions)
- Head 2: Gate (sigmoid, P(acceptable | x))
- Head 3: Head (multi-class, provides auxiliary predictions)

Shared backbone: DINOv3 (frozen, trained in Phase 1)
Backbone extraction: Shared feature extraction from DINOv3

SELECTIVENET OBJECTIVE (Phase 3.3 - PROPER):
L_sel = Σ[i∈g] ⋅ CE(f_i, y_i) + Σ[i∉g] ⋅ CE(0, y_i) + λ ⋅ max(0, c - Σ[i]⋅g_i / N)²

Where:
- g_i = (gate_logits_i > 0) - selection/acceptance (exit when gate is positive)
- CE(f_i, y_i) = cross_entropy(classifier_logits_i, labels_i) - classifier loss
- CE(0, y_i) = cross_entropy(zeros, labels_i) - constant class loss
- c = target_coverage (default 0.5, encourage ~50% coverage)
- λ = coverage penalty weight

Gate Semantics (Option A: P(acceptable | x)):
- Gate output g(x) ∈ [0,1] represents P(acceptable | x)
- "acceptable" is defined by the constrained selective objective
- Exit when calibrated g(x) ≥ τ (threshold from val_calib)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class GateHead(nn.Module):
    """
    Gate Head for selective classification (Phase 3.1) - SELECTIVENET-STYLE.
    
    Architecture: SelectiveNet-style 3-head design
    - Head 1: Classifier (softmax, num_classes=2)
    - Head 2: Gate (sigmoid, P(acceptable | x))
    - Head 3: Head (multi-class, sigmoid)
    
    Gate Semantics (Option A: P(acceptable | x)):
    - Gate output g(x) ∈ [0,1] represents P(acceptable | x)
    - High gate score → exit (sample is acceptable to skip further processing)
    - Low gate score → continue (sample needs full pipeline)
    
    Input: DINOv3 features [N, 768]
    Output: [classifier_logits, gate_logits, head_logits]
    
    Args:
        backbone_dim: DINOv3 feature dimension (768)
        num_classes: Number of classes (default 2)
        gate_hidden_dim: Hidden dimension for gate head (default 128)
        use_ema: Use EMA for gate head (default False)
        device: Device to load model on (default "cuda")
        verbose: Print status messages (default True)
    """
    
    def __init__(
        self,
        backbone_dim: int = 768,
        num_classes: int = 2,
        gate_hidden_dim: int = 128,
        use_ema: bool = False,
        device: str = "cuda",
        verbose: bool = True
    ):
        super(GateHead, self).__init__()
        
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes
        self.gate_hidden_dim = gate_hidden_dim
        self.use_ema = use_ema
        self.device = device
        self.verbose = verbose
        
        # Head 1: Classifier (Phase 1.5)
        # Input: DINOv3 features [N, 768]
        # Output: Logits [N, num_classes]
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),  # Hidden layer
            nn.ReLU(),
            nn.Linear(backbone_dim, num_classes)  # Output layer
        )
        
        # Head 2: Gate (sigmoid, P(acceptable | x))
        # Input: DINOv3 features [N, 768]
        # Output: Gate logits [N, 1] (P(acceptable | x))
        self.gate = nn.Sequential(
            nn.Linear(backbone_dim, gate_hidden_dim),  # Hidden layer
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),  # Output layer (sigmoid)
        )
        
        # Head 3: Head (multi-class, sigmoid)
        # Input: DINOv3 features [N, 768]
        # Output: Head logits [N, num_classes] (auxiliary)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, gate_hidden_dim),  # Hidden layer
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, num_classes)  # Output layer (sigmoid)
        )
        
        # Move to device
        self.classifier = self.classifier.to(device)
        self.gate = self.gate.to(device)
        self.head = self.head.to(device)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PHASE 3.1: GATE HEAD ARCHITECTURE (SELECTIVENET-STYLE)")
            print(f"{'='*80}")
            print(f"   Backbone Dim: {backbone_dim}")
            print(f"   Num Classes: {num_classes}")
            print(f"   Gate Hidden Dim: {gate_hidden_dim}")
            print(f"   Classifier: 768 -> {backbone_dim} -> {num_classes}")
            print(f"   Gate: 768 -> {gate_hidden_dim} -> 1 (sigmoid, P(acceptable|x))")
            print(f"   Head: 768 -> {gate_hidden_dim} -> {num_classes} (multi-class)")
            print(f"   Use EMA: {use_ema}")
            print(f"   Gate Semantics: P(acceptable|x) - Exit when high gate score")
            print(f"{'='*80}")
    
    def forward(
        self,
        features: torch.Tensor,
        return_classifier_logits: bool = False,
        return_gate_logits: bool = False,
        return_head_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through 3-head architecture.
        
        Args:
            features: DINOv3 features [N, backbone_dim]
            return_classifier_logits: Return classifier logits
            return_gate_logits: Return gate logits
            return_head_logits: Return head logits
        
        Returns:
            Dict with keys: 'classifier_logits', 'gate_logits', 'head_logits'
        """
        if self.verbose:
            print(f"\n   3-Head Forward Pass:")
            print(f"   Input shape: {features.shape}")
        
        # Head 1: Classifier
        classifier_logits = self.classifier(features)
        
        # Head 2: Gate (shape: [N, 1] for P(acceptable|x))
        gate_logits = self.gate(features)
        
        # Head 3: Head
        head_logits = self.head(features)
        
        if self.verbose:
            print(f"   Classifier logits: {classifier_logits.shape}")
            print(f"   Gate logits: {gate_logits.shape}")
            print(f"   Head logits: {head_logits.shape}")
        
        outputs = {
            'classifier_logits': classifier_logits,
            'gate_logits': gate_logits,
            'head_logits': head_logits
        }
        
        if return_classifier_logits:
            return outputs['classifier_logits']
        elif return_gate_logits:
            return outputs['gate_logits']
        elif return_head_logits:
            return outputs['head_logits']
        else:
            # Return all outputs (default)
            return outputs
    
    def get_exit_mask_from_score(
        self,
        gate_score: torch.Tensor,
        threshold: float = 0.5,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Compute exit mask from calibrated gate score (Phase 3.11).
        
        Gate Semantics (Option A): Exit when gate_score ≥ τ
        - gate_score ∈ [0,1] represents P(acceptable | x)
        - High score → exit (sample acceptable to skip further processing)
        
        Args:
            gate_score: Calibrated gate score [N, 1]
            threshold: Gate threshold τ (from val_calib)
            verbose: Print status messages
        
        Returns:
            Binary mask [N] (1=exit, 0=continue)
        """
        if verbose:
            print(f"\n   Gate Exit Mask (from calibrated score):")
            print(f"   Gate score: {gate_score.shape}")
            print(f"   Threshold τ: {threshold}")
        
        # Gate score: [N, 1] -> [N]
        gate_score_flat = gate_score.squeeze(1)
        
        # Exit condition: gate_score ≥ τ
        exit_mask = gate_score_flat >= threshold
        
        if verbose:
            exit_count = exit_mask.sum().item()
            print(f"   Exit samples: {exit_count}/{gate_score_flat.shape[0]}")
            print(f"   Exit mask shape: {exit_mask.shape}")
        
        return exit_mask


def compute_selective_loss(
    classifier_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    labels: torch.Tensor,
    target_coverage: float = 0.5,
    coverage_penalty_weight: float = 1.0,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SelectiveNet-style objective (Phase 3.3 & 3.4) - PROPER VERSION.
    
    CRITICAL FIX: Uses SOFT gates g = σ(gate_logits) for differentiable training.
    
    Implements differentiable selective risk + coverage penalty:
    L_sel = (Σ[i] g_i ⋅ CE(f_i, y_i)) / (Σ[i] g_i + ε) + λ ⋅ max(0, c - mean(g))²
    
    Where:
    - g_i = σ(gate_logits_i) - SOFT selection weight in [0,1]
    - CE(f_i, y_i) = cross_entropy(classifier_logits_i, labels_i) - per-sample classifier loss
    - c = target_coverage (default 0.5, encourage ~50% coverage)
    - λ = coverage penalty weight
    - ε = small constant for numerical stability
    
    Gate Semantics: g_i ≈ P(Stage A is correct | x)
    - High g(x) → exit (Stage A is correct/confident)
    - Low g(x) → continue (needs Stage B)
    
    CRITICAL: Gate trains end-to-end with gradients flowing through g = σ(logits)
    
    Args:
        classifier_logits: Classifier logits [N, num_classes]
        gate_logits: Gate logits [N, 1]
        labels: True labels [N]
        target_coverage: Target coverage c (default 0.5)
        coverage_penalty_weight: Weight for coverage penalty λ (default 1.0)
        verbose: Print status messages
    
    Returns:
        (g, selective_loss, coverage) where:
        - g: Soft gate probabilities [N] (for training, differentiable)
        - selective_loss: Total selective loss (scalar)
        - coverage: Mean(g) (fraction accepted, scalar)
    """
    
    eps = 1e-6  # Numerical stability
    
    # CRITICAL FIX (Phase 3.3 & 3.4): Use SOFT gates g = σ(gate_logits) for DIFFERENTIABLE training
    # g_i ∈ [0,1] represents P(Stage A is correct | x)
    g = torch.sigmoid(gate_logits.squeeze(1))  # [N, 1] -> [N]
    
    # Coverage: mean of soft gate probabilities
    coverage = g.mean()
    
    # CRITICAL FIX (Phase 3.4): Per-sample CE loss (using LOGITS, not softmax)
    # CE_i = cross_entropy(classifier_logits_i, labels_i)
    per_sample_ce = nn.functional.cross_entropy(
        classifier_logits,
        labels,
        reduction='none'
    )  # [N]
    
    # CRITICAL FIX (Phase 3.4): Selective risk normalized by sum(g)
    # R_sel = (Σ[i] g_i ⋅ CE_i) / (Σ[i] g_i + ε)
    # This is core SelectiveNet objective: error conditional on selection
    selective_risk = (g * per_sample_ce).sum() / (g.sum() + eps)
    
    # Coverage penalty: λ ⋅ max(0, c - mean(g))²
    # Encourages coverage ≈ target coverage c
    coverage_penalty = coverage_penalty_weight * torch.max(
        torch.tensor(0.0, device=classifier_logits.device),
        target_coverage - coverage
    ) ** 2
    
    # Total selective loss: selective risk + coverage penalty
    selective_loss = selective_risk + coverage_penalty
    
    # Hard selection mask for inference/metrics (NOT for training gradients)
    selection_mask = (g >= 0.5).float()  # [N]: 1=exit, 0=continue
    
    if verbose:
        print(f"   Soft gates g: {g.shape}, mean={coverage.item():.4f}")
        print(f"   Per-sample CE: {per_sample_ce.shape}, mean={per_sample_ce.mean().item():.6f}")
        print(f"   Selective risk: {selective_risk.item():.6f}")
        print(f"   Coverage penalty: {coverage_penalty.item():.6f}")
        print(f"   Selective loss: {selective_loss.item():.6f}")
        print(f"   Selection mask (for inference): {selection_mask.shape}, mean={selection_mask.mean().item():.4f}")
    
    return selection_mask, selective_loss, coverage

def compute_auxiliary_loss(
    head_logits: torch.Tensor,
    labels: torch.Tensor,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute auxiliary loss (Phase 3.4) - PROPER VERSION.
    
    Trains head head to improve classifier accuracy.
    Auxiliary loss: Cross-entropy on head outputs (LOGITS, not softmax).
    
    Args:
        head_logits: Head logits [N, num_classes]
        labels: True labels [N]
        verbose: Print status messages
    
    Returns:
        Auxiliary loss (scalar)
    """
    # PROPER: Cross-entropy expects LOGITS, NOT softmax probabilities
    head_loss = nn.functional.cross_entropy(head_logits, labels, reduction='mean')
    
    if verbose:
        print(f"   Head Loss: {head_loss.item():.6f}")
    
    return head_loss
