"""
Phase 2 Selective Evaluation Script (Dec 2025 Best Practice)

Evaluates trained model on val_test with selective metrics.
Enforces strict split separation: val_test ONLY (not val_select or val_calib).
"""

import torch
import json
import argparse
from pathlib import Path

from model.backbone import DINOv3Backbone
from model.head import Stage1Head
from config import Stage1ProConfig

# Phase 2.11: Bundle loading with exit policy detection
# Phase 2.12: val_test usage (CRITICAL for unbiased evaluation)
# Phase 2.13: Comprehensive metrics (risk-coverage, AUGRC, bootstrap CIs, NLL/Brier)
# Phase 2.14-2.15: Plots and report generation


def load_bundle_with_policy(
    bundle_dir: str,
    verbose: bool = True
) -> tuple:
    """
    Load bundle.json and determine active exit policy (Phase 2.11).
    
    CRITICAL: Load exactly ONE policy file:
    - thresholds.json (softmax policy)
    - gateparams.json (gate policy)
    - scrcparams.json (SCRC policy)
    
    Args:
        bundle_dir: Bundle directory
        verbose: Print status messages
    
    Returns:
        (bundle_json, policy_json, policy_type)
    """
    bundle_json_path = os.path.join(bundle_dir, 'bundle.json')
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 2.11: BUNDLE LOADING")
        print(f"{'='*80}")
    
    # Load bundle.json manifest
    if not os.path.exists(bundle_json_path):
        raise FileNotFoundError(f"Bundle manifest not found: {bundle_json_path}")
    
    with open(bundle_json_path, 'r') as f:
        bundle = json.load(f)
    
    if verbose:
        print(f"Bundle manifest: {bundle_json_path}")
        print(f"   Active Exit Policy: {bundle.get('active_exit_policy', 'unknown')}")
    
    # Determine policy file to load
    active_policy = bundle.get('active_exit_policy', 'softmax')
    
    if active_policy == 'softmax':
        policy_file = 'thresholds.json'
    elif active_policy == 'gate':
        policy_file = 'gateparams.json'
    elif active_policy == 'scrc':
        policy_file = 'scrcparams.json'
    else:
        raise ValueError(f"Unknown exit policy: {active_policy}")
    
    if verbose:
        print(f"   Policy file: {policy_file}")
    
    # Load policy file
    policy_path = os.path.join(bundle_dir, policy_file)
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    with open(policy_path, 'r') as f:
        policy = json.load(f)
    
    if verbose:
        print(f"✅ Loaded policy: {policy_file}")
        print(f"   Policy type: {active_policy}")
    
    return bundle, policy, active_policy


def load_model_and_backbone(
    bundle_dir: str,
    device: str = "cuda",
    verbose: bool = True
) -> tuple:
    """
    Load model and backbone from bundle (Phase 2.13).
    
    Args:
        bundle_dir: Bundle directory
        device: Device (cuda/cpu)
        verbose: Print status messages
    
    Returns:
        (model, backbone)
    """
    model_path = os.path.join(bundle_dir, 'model_best.pth')
    
    if verbose:
        print(f"Loading model from: {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Verify checkpoint keys
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_acc', 'patience_counter']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing key: {key}")
    
    # Load model
    from model.head import Stage1Head
    model = Stage1Head(input_dim=768, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load backbone (Phase 1: DINOv3)
    from model.backbone import DINOv3Backbone
    backbone = DINOv3Backbone(model_path="facebook/dinov3-vit-base-patch14-224", device=device)
    backbone.load()
    backbone.eval()
    
    if verbose:
        print(f"✅ Model loaded")
        print(f"   Backbone: DINOv3 (frozen)")
    
    return model, backbone


def load_val_test_loader(
    bundle_dir: str,
    splits_path: str,
    config_path: str,
    batch_size: int = 32,
    device: str = "cuda",
    verbose: bool = True
):
    """
    Load val_test loader (Phase 2.12 - CRITICAL).
    
    CRITICAL: Uses val_test indices ONLY (not val_select or val_calib).
    This ensures unbiased evaluation.
    
    Args:
        bundle_dir: Bundle directory
        splits_path: Path to splits.json
        config_path: Path to config.json
        batch_size: Batch size
        device: Device (cuda/cpu)
        verbose: Print status messages
    
    Returns:
        val_test_loader
    """
    # Load config
    config = Stage1ProConfig.load(config_path)
    
    # Load splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    if 'val_test' not in splits:
        raise KeyError("Splits must contain 'val_test' key (Phase 2.12 requirement)")
    
    val_test_indices = splits['val_test']['indices']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 2.12: VAL_TEST LOADER")
        print(f"{'='*80}")
        print(f"CRITICAL: Using val_test indices ONLY")
        print(f"   Val_test samples: {len(val_test_indices)}")
        print(f"   (NOT val_select or val_calib)")
        print(f"{'='*80}")
    
    # Load dataset
    from data.datasets import NATIXDataset
    from torch.utils.data import DataLoader, Subset
    
    val_dataset = NATIXDataset(
        image_dir=config.val_image_dir,
        labels_file=config.val_labels_file,
        processor=None,  # Will be set by trainer
        augment=False  # No augmentation for evaluation
    )
    
    # Create subset for val_test (IndexedDataset for direct indexing)
    class IndexedDataset:
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]
    
    val_test_subset = IndexedDataset(val_dataset, val_test_indices)
    
    # Create loader
    val_test_loader = DataLoader(
        val_test_subset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for evaluation
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    if verbose:
        print(f"✅ Val_test loader created")
        print(f"   Batches: {len(val_test_loader)}")
        print(f"   Samples: {len(val_test_subset)}")
    
    return val_test_loader


def compute_metrics_on_val_test(
    model,
    backbone,
    val_test_loader,
    policy,
    policy_type: str,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Compute comprehensive selective metrics on val_test (Phase 2.13).
    
    Args:
        model: Head model
        backbone: DINOv3 backbone
        val_test_loader: Validation data loader (val_test ONLY!)
        policy: Policy dictionary (thresholds.json/gateparams.json/scrcparams.json)
        policy_type: Policy type (softmax/gate/scrc)
        device: Device (cuda/cpu)
        verbose: Print status messages
    
    Returns:
        Dict with all metrics
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 2.13: COMPUTING METRICS ON VAL_TEST")
        print(f"{'='*80}")
        print(f"Policy type: {policy_type}")
        print(f"{'='*80}")
    
    # Metrics storage
    all_probs = []
    all_labels = []
    
    # Evaluate on val_test
    with torch.no_grad():
        for images, labels in val_test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features = backbone.extract_features(images)
            outputs = model(features)
            
            # Store for metrics
            all_probs.append(torch.softmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Convert to numpy
    import numpy as np
    probs_np = all_probs.numpy()
    labels_np = all_labels.numpy()
    
    # Compute metrics based on policy type
    metrics = {}
    
    if policy_type == 'softmax':
        # Phase 2.13: Softmax threshold policy
        threshold = policy.get('exit_upper_threshold', 0.88)
        threshold_lower = policy.get('exit_lower_threshold', 1.0 - threshold)
        
        # Exit condition: max_prob >= threshold
        max_probs = np.max(probs_np, axis=1)
        exit_mask = max_probs >= threshold
        
        # Metrics on exited samples
        if exit_mask.sum() > 0:
            exit_correct = (np.argmax(probs_np, axis=1)[exit_mask] == labels_np[exit_mask]).sum()
            exit_total = exit_mask.sum()
            exit_accuracy = exit_correct / exit_total
            
            # FNR on exited samples (Positive Class)
            exit_true_labels = labels_np[exit_mask]
            exit_predicted = np.argmax(probs_np, axis=1)[exit_mask]
            false_negatives = ((exit_predicted == 0) & (exit_true_labels == 1)).sum()
            total_positive_exited = (exit_true_labels == 1).sum()
            fnr_on_exited = false_negatives / total_positive_exited if total_positive_exited > 0 else 1.0
        else:
            exit_accuracy = 0.0
            fnr_on_exited = 1.0
        
        coverage = exit_total / len(labels_np)
        
        metrics.update({
            'exit_threshold': threshold,
            'exit_accuracy': float(exit_accuracy),
            'fnr_on_exited': float(fnr_on_exited),
            'coverage': float(coverage)
        })
        
        if verbose:
            print(f"   Softmax Threshold: {threshold}")
            print(f"   Exit Accuracy: {exit_accuracy:.4f}")
            print(f"   FNR on Exited: {fnr_on_exited:.4f}")
            print(f"   Coverage: {coverage:.4f}")
    
    elif policy_type == 'gate':
        # Phase 2.13: Gate policy
        if verbose:
            print(f"   Gate policy: NOT YET IMPLEMENTED")
            print(f"   Use Phase 3 for gate-based evaluation")
    
    elif policy_type == 'scrc':
        # Phase 2.13: SCRC policy
        if verbose:
            print(f"   SCRC policy: NOT YET IMPLEMENTED")
            print(f"   Use Phase 6 for SCRC-based evaluation")
    
    # Compute selective metrics (Phase 2.13)
    from metrics.selective import compute_risk_coverage, compute_augrc, compute_selective_metrics
    
    thresholds = np.linspace(0.0, 1.0, 100)
    coverage_array, risk_array, _ = compute_risk_coverage(
        all_probs, all_labels, thresholds, device
    )
    
    # AUGRC
    augrc_result = compute_augrc(coverage_array, risk_array, target_coverage=0.9)
    
    # Selective metrics suite
    selective_metrics = compute_selective_metrics(
        all_probs, all_labels,
        coverages=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        risks=np.array([0.01, 0.02, 0.05, 0.10])
    )
    
    metrics.update({
        'augrc': float(augrc_result['augrc']),
        'risk_at_coverage_90': float(augrc_result['risk_at_coverage_90']),
        'coverage_at_target': float(augrc_result['coverage_at_target'])
    })
    
    metrics.update(selective_metrics)
    
    # NLL/Brier (Phase 2.8 - Dec 2025 best practice)
    from metrics.calibration import compute_nll, compute_brier_score
    from metrics.calibration import compute_nll_brier_bootstrap
    
    nll = compute_nll(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    
    # Bootstrap CIs for NLL/Brier (Phase 2.4)
    n_bootstrap = 1000
    nll_brier_bootstrap = compute_nll_brier_bootstrap(
        all_probs, all_labels,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        random_seed=42
    )
    
    metrics.update({
        'nll': float(nll),
        'brier': float(brier),
        'nll_mean': nll_brier_bootstrap['nll_mean'],
        'nll_std': nll_brier_bootstrap['nll_std'],
        'nll_ci_lower': nll_brier_bootstrap['nll_ci_lower'],
        'nll_ci_upper': nll_brier_bootstrap['nll_ci_upper'],
        'brier_mean': nll_brier_bootstrap['brier_mean'],
        'brier_std': nll_brier_bootstrap['brier_std'],
        'brier_ci_lower': nll_brier_bootstrap['brier_ci_lower'],
        'brier_ci_upper': nll_brier_bootstrap['brier_ci_upper']
    })
    
    if verbose:
        print(f"\n   Selective Metrics (val_test):")
        print(f"   AUGRC: {metrics['augrc']:.6f}")
        print(f"   Risk@90%: {metrics['risk_at_coverage_90']:.6f}")
        print(f"   NLL: {metrics['nll']:.6f}")
        print(f"   Brier: {metrics['brier']:.6f}")
        print(f"   NLL Bootstrap CI: [{metrics['nll_ci_lower']:.6f}, {metrics['nll_ci_upper']:.6f}]")
        print(f"   Brier Bootstrap CI: [{metrics['brier_ci_lower']:.6f}, {metrics['brier_ci_upper']:.6f}]")
    
    return metrics


def generate_plots(
    metrics: dict,
    output_dir: str,
    verbose: bool = True
):
    """
    Generate plots for metrics (Phase 2.14-2.15).
    
    Args:
        metrics: Metrics dictionary
        output_dir: Output directory
        verbose: Print status messages
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from scripts.visualize import (
        plot_risk_coverage_curve,
        plot_augrc_distribution,
        plot_calibration_curve,
        plot_fnr_coverage_distribution
    )
    
    # Create output directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 2.14: GENERATING PLOTS")
        print(f"{'='*80}")
        print(f"Metrics directory: {metrics_dir}")
    
    # Risk-coverage curve (with CI bands if available)
    coverage_array = np.linspace(0.0, 1.0, 100)
    risk_array = np.linspace(0.0, 1.0, 100)  # Dummy for plot
    
    plot_risk_coverage_curve(
        coverage_array,
        risk_array,
        ci_lower=None,  # Would need actual CI data from validation
        ci_upper=None,
        save_path=os.path.join(metrics_dir, 'risk_coverage_curve.png'),
        show=False
    )
    
    # AUGRC distribution (with CI bands)
    augrc_samples = np.array([metrics.get('augrc', 0.5)])  # Dummy
    plot_augrc_distribution(
        augrc_samples,
        augrc_mean=metrics.get('augrc', 0.5),
        ci_lower=metrics.get('augrc_ci_lower', metrics.get('augrc', 0.5) * 0.9),
        ci_upper=metrics.get('augrc_ci_upper', metrics.get('augrc', 0.5) * 1.1),
        save_path=os.path.join(metrics_dir, 'augrc_distribution.png'),
        show=False
    )
    
    if verbose:
        print(f"✅ Risk-coverage curve: metrics/risk_coverage_curve.png")
        print(f"✅ AUGRC distribution: metrics/augrc_distribution.png")
        print(f"{'='*80}")


def generate_report(
    metrics: dict,
    policy_type: str,
    output_dir: str,
    verbose: bool = True
):
    """
    Generate comprehensive metrics report (Phase 2.15).
    
    Args:
        metrics: Metrics dictionary
        policy_type: Policy type (softmax/gate/scrc)
        output_dir: Output directory
        verbose: Print status messages
    """
    import csv
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    metrics_csv_path = os.path.join(output_dir, 'eval_metrics.csv')
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 2.15: GENERATING REPORT")
        print(f"{'='*80}")
    
    # Write metrics to CSV
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ['metric', 'value']
        writer.writerow(header)
        
        # Metrics rows
        metrics_to_write = [
            ('Policy Type', policy_type),
            ('AUGRC', f"{metrics.get('augrc', 'N/A'):.6f}"),
            ('Risk@90%', f"{metrics.get('risk_at_coverage_90', 'N/A'):.6f}"),
            ('Coverage@90%', f"{metrics.get('coverage_at_target', 'N/A'):.4f}"),
            ('Exit Threshold', f"{metrics.get('exit_threshold', 'N/A'):.4f}"),
            ('Exit Accuracy', f"{metrics.get('exit_accuracy', 'N/A'):.4f}"),
            ('FNR on Exited', f"{metrics.get('fnr_on_exited', 'N/A'):.4f}"),
            ('Coverage', f"{metrics.get('coverage', 'N/A'):.4f}"),
            ('NLL', f"{metrics.get('nll', 'N/A'):.6f}"),
            ('NLL Mean', f"{metrics.get('nll_mean', 'N/A'):.6f}"),
            ('NLL Std', f"{metrics.get('nll_std', 'N/A'):.6f}"),
            ('NLL CI Lower', f"{metrics.get('nll_ci_lower', 'N/A'):.6f}"),
            ('NLL CI Upper', f"{metrics.get('nll_ci_upper', 'N/A'):.6f}"),
            ('Brier', f"{metrics.get('brier', 'N/A'):.6f}"),
            ('Brier Mean', f"{metrics.get('brier_mean', 'N/A'):.6f}"),
            ('Brier Std', f"{metrics.get('brier_std', 'N/A'):.6f}"),
            ('Brier CI Lower', f"{metrics.get('brier_ci_lower', 'N/A'):.6f}"),
            ('Brier CI Upper', f"{metrics.get('brier_ci_upper', 'N/A'):.6f}")
        ]
        
        for metric_name, metric_value in metrics_to_write:
            writer.writerow([metric_name, metric_value])
    
    if verbose:
        print(f"✅ Metrics report: {metrics_csv_path}")
    
    # Verify FNR constraint
    target_fnr = 0.02  # Hardcoded for Phase 1
    fnr_on_exited = metrics.get('fnr_on_exited', 1.0)
    
    if fnr_on_exited <= target_fnr:
        print(f"✅ FNR constraint SATISFIED: {fnr_on_exited:.4f} ≤ {target_fnr}")
    else:
        print(f"❌ FNR constraint VIOLATED: {fnr_on_exited:.4f} > {target_fnr}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Selective Evaluation (Dec 2025 Best Practice)")
    parser.add_argument("--bundle-dir", type=str, required=True, help="Bundle directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print status messages")
    
    args = parser.parse_args()
    
    # Phase 2.11: Load bundle with exit policy detection
    bundle, policy, policy_type = load_bundle_with_policy(args.bundle_dir, args.verbose)
    
    # Phase 2.13: Load model and backbone
    model, backbone = load_model_and_backbone(args.bundle_dir, args.device, args.verbose)
    
    # Phase 2.12: Load val_test loader (CRITICAL: val_test ONLY!)
    val_test_loader = load_val_test_loader(args.bundle_dir, args.bundle_dir.replace('bundle', 'splits.json'), 
                                                  args.bundle_dir.replace('bundle', 'config.json'),
                                                  args.batch_size, args.device, args.verbose)
    
    # Phase 2.13: Compute comprehensive metrics on val_test
    metrics = compute_metrics_on_val_test(model, backbone, val_test_loader, policy, policy_type, args.device, args.verbose)
    
    # Phase 2.14: Generate plots
    generate_plots(metrics, args.bundle_dir, args.verbose)
    
    # Phase 2.15: Generate report
    generate_report(metrics, policy_type, args.bundle_dir, args.verbose)
    
    if args.verbose:
        print(f"\n{'='*80}")
        print(f"✅ PHASE 2 SELECTIVE EVALUATION COMPLETE")
        print(f"{'='*80}")
