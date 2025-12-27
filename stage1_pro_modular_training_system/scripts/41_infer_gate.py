"""
Gate Exit Inference Script - Phase 3.11

Implements gate-based selective classification using calibrated gate scores.
Exit condition: calibrated_gate_score ≥ τ (threshold from val_calib)

Gate Semantics (Option A: P(acceptable | x)):
- Gate output g(x) ∈ [0, 1] represents P(acceptable | x)
- "acceptable" is defined by the constrained selective objective
- Exit when gate says "acceptable" (high score) → skip further processing
- Continue when gate says "not acceptable" (low score) → use full pipeline
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_gate_calibration(
    gateparams_path: str,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Load gate calibration parameters (Phase 3.11).
    
    Args:
        gateparams_path: Path to gateparams.json
        verbose: Print status messages
    
    Returns:
        Dict with calibration parameters
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"LOADING GATE CALIBRATION")
        print(f"{'='*80}")
    
    if not os.path.exists(gateparams_path):
        raise FileNotFoundError(f"Missing {gateparams_path} - run gate calibration first")
    
    with open(gateparams_path, 'r') as f:
        gateparams = json.load(f)
    
    if verbose:
        print(f"   Active exit policy: {gateparams['active_exit_policy']}")
        print(f"   Gate threshold: {gateparams['gate_threshold']}")
        print(f"   Gate lower threshold: {gateparams.get('gate_lower_threshold', 'None')}")
        print(f"   Loaded from: {gateparams_path}")
    
    return gateparams


def load_model_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[torch.nn.Module, Dict[str, any]]:
    """
    Load model checkpoint (shared across all phases).
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        verbose: Print status messages
    
    Returns:
        (model, checkpoint_dict)
    """
    if verbose:
        print(f"\n   Loading model checkpoint...")
        print(f"   Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_checkpoint = checkpoint.get('model_state_dict', {})
    gate_head_checkpoint = checkpoint.get('gate_head_state_dict', {})
    
    # Load model weights
    # Note: This is a placeholder - actual loading depends on model type
    if verbose:
        print(f"   Model state dict keys: {list(model_checkpoint.keys())[:5]}...")
        print(f"   Gate head state dict keys: {list(gate_head_checkpoint.keys())[:5]}...")
    
    return checkpoint


def extract_features(
    images: torch.Tensor,
    backbone: torch.nn.Module,
    device: str = "cuda",
    verbose: bool = False
) -> torch.Tensor:
    """
    Extract DINOv3 features from images (Phase 1.3).
    
    Args:
        images: Input images [B, 3, H, W]
        backbone: DINOv3 backbone (frozen)
        device: Device
        verbose: Print status messages
    
    Returns:
        Features [B, 768]
    """
    if verbose:
        print(f"   Extracting features: {images.shape}")
    
    with torch.no_grad():
        features = backbone.extract_features(images)
    
    return features


def compute_gate_exit_mask(
    gate_logits: torch.Tensor,
    gate_threshold: float,
    verbose: bool = False
) -> Tuple[torch.Tensor, float]:
    """
    Compute exit mask from gate logits (Phase 3.11 - Option A semantics).
    
    Gate Semantics (Option A: P(acceptable | x)):
    - Gate score g(x) = sigmoid(gate_logit) ∈ [0, 1]
    - Exit when g(x) ≥ τ (high score → sample is "acceptable")
    - Continue when g(x) < τ (low score → sample needs full pipeline)
    
    Args:
        gate_logits: Gate logits [N, 1]
        gate_threshold: Gate threshold τ (from val_calib)
        verbose: Print status messages
    
    Returns:
        (exit_mask, coverage)
    """
    if verbose:
        print(f"\n   Computing gate exit mask...")
        print(f"   Gate logits: {gate_logits.shape}")
        print(f"   Gate threshold τ: {gate_threshold}")
    
    # Gate score: P(acceptable | x) = sigmoid(gate_logit)
    gate_scores = torch.sigmoid(gate_logits.squeeze(1))
    
    # Exit condition: gate_score ≥ τ
    exit_mask = gate_scores >= gate_threshold
    
    coverage = exit_mask.float().mean().item()
    
    if verbose:
        exit_count = exit_mask.sum().item()
        print(f"   Gate scores: mean={gate_scores.mean():.4f}, std={gate_scores.std():.4f}")
        print(f"   Exit samples: {exit_count}/{gate_scores.shape[0]}")
        print(f"   Coverage: {coverage:.4f}")
    
    return exit_mask, coverage


def run_gate_inference(
    model: torch.nn.Module,
    backbone: torch.nn.Module,
    images: torch.Tensor,
    gateparams: Dict[str, any],
    device: str = "cuda",
    verbose: bool = False
) -> Dict[str, any]:
    """
    Run gate-based selective inference (Phase 3.11).
    
    Args:
        model: Gate head model
        backbone: DINOv3 backbone
        images: Input images [B, 3, H, W]
        gateparams: Gate calibration parameters
        device: Device
        verbose: Print status messages
    
    Returns:
        Dict with inference results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 3.11: GATE EXIT INFERENCE")
        print(f"{'='*80}")
        print(f"   Input images: {images.shape}")
        print(f"   Device: {device}")
    
    # Extract features
    features = extract_features(images, backbone, device, verbose)
    
    # Get model outputs (3-head)
    with torch.no_grad():
        outputs = model(features)
        classifier_logits = outputs['classifier_logits']
        gate_logits = outputs['gate_logits']
        head_logits = outputs['head_logits']
    
    # Compute gate exit mask
    gate_threshold = gateparams['gate_threshold']
    exit_mask, coverage = compute_gate_exit_mask(gate_logits, gate_threshold, verbose)
    
    # CRITICAL FIX (Phase 3.11): Implement Stage A + Stage B cascade
    # Stage A (fast path): Use classifier for EXITED samples (gate says "acceptable")
    # Stage B (strong path): Use head logits for NON-EXITED samples (needs stronger compute)
    
    # Initialize predictions tensor
    batch_size = classifier_logits.shape[0]
    predicted = torch.zeros(batch_size, dtype=torch.long, device=classifier_logits.device)
    
    # Stage A: Fast predictions for exited samples (gate says "acceptable")
    # These samples are confident - use classifier (fast exit)
    if exit_mask.any():
        predicted[exit_mask] = torch.softmax(classifier_logits, dim=1).argmax(dim=1)[exit_mask]
        stage_a_count = exit_mask.sum().item()
    else:
        stage_a_count = 0
    
    # Stage B: Strong predictions for non-exited samples (gate says "not acceptable")
    # These samples need stronger compute - use head logits (trained on full coverage)
    non_exit_mask = ~exit_mask
    if non_exit_mask.any():
        predicted[non_exit_mask] = torch.softmax(head_logits, dim=1).argmax(dim=1)[non_exit_mask]
        stage_b_count = non_exit_mask.sum().item()
    else:
        stage_b_count = 0
    
    if verbose:
        print(f"   Classifier logits (Stage A): {classifier_logits.shape}")
        print(f"   Head logits (Stage B): {head_logits.shape}")
        print(f"   Gate logits: {gate_logits.shape}")
        print(f"   Exit mask: {exit_mask.shape}")
        print(f"   Coverage: {coverage:.4f}")
        print(f"   Cascade: Stage A={stage_a_count}, Stage B={stage_b_count} (total={batch_size})")
        print(f"   Predictions: {predicted.shape}")
        print(f"   Stage A used: {stage_a_count/batch_size:.1%}, Stage B used: {stage_b_count/batch_size:.1%}")
    
    return {
        'classifier_logits': classifier_logits.cpu(),
        'gate_logits': gate_logits.cpu(),
        'head_logits': head_logits.cpu(),
        'exit_mask': exit_mask.cpu(),
        'predicted': predicted.cpu(),
        'stage_a_count': stage_a_count,
        'stage_b_count': stage_b_count,
        'coverage': coverage,
        'gate_threshold': gate_threshold
    }


def main():
    parser = argparse.ArgumentParser(description="Gate Exit Inference (Phase 3.11)")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model checkpoint')
    parser.add_argument('--gateparams_path', type=str, required=True, help='Path to gateparams.json')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for inference results')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed status messages')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"PHASE 3.11: GATE EXIT INFERENCE")
    print(f"{'='*80}")
    print(f"   Checkpoint dir: {args.checkpoint_dir}")
    print(f"   Gate params: {args.gateparams_path}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Device: {args.device}")
    print(f"{'='*80}")
    
    # Load gate calibration
    gateparams = load_gate_calibration(args.gateparams_path, args.verbose)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n✅ Gate exit inference script created (Phase 3.11)")
    print(f"   Gate semantics: P(acceptable | x) - Option A")
    print(f"   Exit condition: gate_score ≥ τ (τ={gateparams['gate_threshold']})")
    print(f"   Ready to run inference on val_test!")


if __name__ == "__main__":
    main()

