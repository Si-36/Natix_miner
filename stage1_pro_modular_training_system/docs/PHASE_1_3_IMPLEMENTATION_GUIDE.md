# Phase 1-3 Implementation Guide (Dec 2025 Best Practices)

## Overview
This guide provides detailed implementation instructions for Phase 1 (Baseline Training), Phase 2 (Selective Evaluation), and Phase 3 (Gate Head) using Dec 2025 state-of-the-art libraries and best practices.

**Key Principle**: Use library-first architecture - leverage HuggingFace Transformers, PEFT, timm, scikit-learn instead of custom implementations.

## Phase 1: Baseline Training (10/10 Todos)

### Phase 1.1: Research DINOv3 Baseline (Dec 2025)

**Research Findings**:
- **HuggingFace Transformers 4.40.0+** provides `AutoModel.from_pretrained()` for loading DINOv3
- **DINOv3-ViT-H/16+**: Latest DINOv3 variant with improved performance (Dec 2025)
- **AutoImageProcessor**: Automatic image normalization (ImageNet stats)
- **PEFT Integration**: `get_peft_model()` available for DoRA/LoRA (though disabled in Phase 1)

**Implementation Checklist**:
- [ ] Use `transformers.AutoModel.from_pretrained()` to load DINOv3 backbone
- [ ] Use `transformers.AutoImageProcessor.from_pretrained()` for image preprocessing
- [ ] Freeze all backbone parameters (except head) using `.requires_grad = False`
- [ ] Extract CLS token from `outputs.last_hidden_state[:, 0, :]`
- [ ] Support torch.compile() for 40% speedup (PyTorch 3.0+)

**Code Example**:
```python
from transformers import AutoModel, AutoImageProcessor

# Load DINOv3 backbone (Dec 2025 best practice)
backbone = AutoModel.from_pretrained(
    config.model_path,  # e.g., "facebook/dinov3-vit-huge-14-224"
    trust_remote_code=True
)
processor = AutoImageProcessor.from_pretrained(config.model_path)

# Freeze backbone (Phase 1: only head trains)
for param in backbone.parameters():
    param.requires_grad = False

# Compile for speedup (PyTorch 3.0+)
head = torch.compile(head, mode="reduce-overhead")
```

### Phase 1.2: Full Reproducibility Implementation (Dec 2025)

**Research Findings**:
- **Deterministic Algorithms**: PyTorch 2.1+ supports deterministic operations
- **Seed Ordering**: Set seeds BEFORE any other imports for consistent RNG state
- **TF32 Precision**: `torch.set_float32_matmul_precision('high')` for faster training without precision loss

**Implementation Checklist**:
- [ ] Set seeds BEFORE imports: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
- [ ] Enable deterministic mode: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- [ ] Enable TF32: `torch.set_float32_matmul_precision('high')`
- [ ] Enable TF32 in backends: `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`
- [ ] Save all seeds to config.json (training seed, data split seed, model seed)

**Code Example**:
```python
import random
import numpy as np
import torch

# Phase 1.2: Full reproducibility (Dec 2025 best practice)
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 precision (faster without significant precision loss)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### Phase 1.3: Complete OOM Error Handling (Dec 2025)

**Research Findings**:
- **Dynamic Batch Size**: Automatically find max batch size that fits in VRAM
- **Graceful Degradation**: Fallback to smaller batch size, not crash
- **GPU Memory Tracking**: Monitor `torch.cuda.memory_allocated()` for debugging

**Implementation Checklist**:
- [ ] Implement `pick_batch_size()` in `data/loaders.py`
- [ ] Try max_batch_size with dummy tensor on GPU
- [ ] Catch `RuntimeError` with 'out of memory' message
- [ ] Fallback to fallback_batch_size on OOM
- [ ] Clear CUDA cache: `torch.cuda.empty_cache()`
- [ ] Provide helpful error messages with recommended batch size

**Code Example**:
```python
# Phase 1.3: OOM error handling (Dec 2025 best practice)
def pick_batch_size(model, input_shape=(3, 224, 224), max_batch_size=64, fallback_batch_size=16, device="cuda"):
    """
    Dynamically find largest batch size that fits in GPU memory.
    
    Args:
        model: Model to test (must be on device)
        input_shape: Input tensor shape (C, H, W)
        max_batch_size: Maximum batch size to try
        fallback_batch_size: Fallback batch size if OOM
        device: Device (cuda/cpu)
    
    Returns:
        Maximum batch size that fits in memory
    """
    batch_size = max_batch_size
    
    while batch_size > fallback_batch_size:
        try:
            # Create dummy input on device
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # If no OOM, this batch size works
            return batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # OOM - reduce batch size and clear cache
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
            else:
                # Other error - re-raise
                raise e
```

### Phase 1.4: Comprehensive Checkpoint Validation (Dec 2025)

**Research Findings**:
- **Robust Loading**: Handle missing keys gracefully
- **Version Compatibility**: Support checkpoint versioning
- **State Restoration**: Load model, optimizer, scheduler states correctly

**Implementation Checklist**:
- [ ] Validate checkpoint file exists
- [ ] Check required keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_acc`, `patience_counter`, `ema_state_dict`
- [ ] Handle missing keys gracefully (skip, not crash)
- [ ] Load EMA shadow state dict (if EMA enabled)
- [ ] Provide recovery options (reset, partial load)

**Code Example**:
```python
# Phase 1.4: Checkpoint validation (Dec 2025 best practice)
def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler, ema):
    """
    Load checkpoint with comprehensive validation and graceful degradation.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state dict into
        optimizer: Optimizer to load state dict into
        scheduler: Scheduler to load state dict into
        ema: EMA object to load shadow state (optional)
    
    Returns:
        dict: Checkpoint data with epoch, best_acc, patience_counter
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Validate required keys
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_acc', 'patience_counter']
    for key in required_keys:
        if key not in checkpoint:
            print(f"⚠️  Missing checkpoint key: {key}")
    
    # Load model state (handle missing gracefully)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️  No model_state_dict in checkpoint. Starting from scratch.")
    
    # Load optimizer state (handle missing gracefully)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("⚠️  No optimizer_state_dict in checkpoint. Starting from scratch.")
    
    # Load scheduler state (handle missing gracefully)
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("⚠️  No scheduler_state_dict in checkpoint. Starting from scratch.")
    
    # Load EMA shadow (handle missing gracefully)
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.shadow = checkpoint['ema_state_dict']
    
    return checkpoint
```

### Phase 1.5: Strict val_select Usage (Dec 2025)

**Research Findings**:
- **Leakage-Free Evaluation**: Use val_select for model selection ONLY
- **val_calib**: Calibration/policy fitting ONLY
- **val_test**: Final unbiased evaluation ONLY
- **Hash-Based Splits**: Deterministic, reproducible

**Implementation Checklist**:
- [ ] Load `splits.json` in trainer
- [ ] Use `val_select` indices for validation dataset
- [ ] NEVER use `val_calib` or `val_test` for model selection
- [ ] Log validation split name ("val_select") in training logs
- [ ] Save validation logits to `val_logits.pt` and `val_labels.pt` for Phase 1 threshold sweep

**Code Example**:
```python
# Phase 1.5: Strict val_select usage (Dec 2025 best practice)
def create_dataloaders(config, splits):
    """
    Create data loaders with strict split separation.
    
    Args:
        config: Training configuration
        splits: Splits dictionary with indices
    
    Returns:
        train_loader, val_select_loader (NOT val_calib!)
    """
    # Load splits
    if os.path.exists(splits_path):
        splits = json.load(open(splits_path))
    else:
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    
    # Create subsets
    train_indices = splits['train']['indices']
    val_select_indices = splits['val_select']['indices']  # ONLY for model selection
    
    # Create datasets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_select_dataset = Subset(full_val_dataset, val_select_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.max_batch_size, shuffle=True)
    val_select_loader = DataLoader(val_select_dataset, batch_size=config.max_batch_size, shuffle=False)
    
    return train_loader, val_select_loader
```

### Phase 1.6: Save All Validation Logits (Dec 2025)

**Research Findings**:
- **Tensor Persistence**: Save logits as PyTorch tensors for Phase 2 threshold sweep
- **Validation Split**: Use val_select for logits (NOT val_calib)
- **Pickle Protocol**: Use `torch.save()` with pickle protocol

**Implementation Checklist**:
- [ ] Collect all logits during validation on val_select
- [ ] Collect all labels during validation
- [ ] Save to `val_logits.pt` in `output_dir`
- [ ] Save to `val_labels.pt` in `output_dir`
- [ ] Use `torch.save()` with pickle protocol

**Code Example**:
```python
# Phase 1.6: Save validation logits (Dec 2025 best practice)
def save_validation_logits(logits_list, labels_list, output_dir):
    """
    Save validation logits for Phase 2 threshold sweep.
    
    Args:
        logits_list: List of logits tensors from validation
        labels_list: List of label tensors from validation
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Concatenate all batches
    all_logits = torch.cat(logits_list, dim=0)  # [N, 2]
    all_labels = torch.cat(labels_list, dim=0)  # [N]
    
    # Save to files
    torch.save(all_logits, os.path.join(output_dir, "val_logits.pt"), pickle_protocol=4)
    torch.save(all_labels, os.path.join(output_dir, "val_labels.pt"), pickle_protocol=4)
    
    print(f"✅ Saved validation logits: {all_logits.shape[0]} samples")
```

### Phase 1.7: Threshold Sweep on val_calib ONLY (Dec 2025)

**Research Findings**:
- **Critical Rule**: Threshold sweep MUST use val_calib (NOT val_select, NOT val_test)
- **Objective**: Maximize coverage subject to FNR ≤ 2%
- **Thresholds to Test**: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]

**Implementation Checklist**:
- [ ] Load `splits.json` and use `val_calib` indices
- [ ] Load validation logits from Phase 1
- [ ] Sweep thresholds: compute exit coverage, exit accuracy, FNR_on_exited
- [ ] Find threshold maximizing coverage with FNR ≤ 2%
- [ ] Save to `thresholds.json` (Phase 1 policy artifact)
- [ ] Validate against `thresholds.schema.json` using `jsonschema`

**Code Example**:
```python
# Phase 1.7: Threshold sweep on val_calib (Dec 2025 best practice)
def threshold_sweep(logits, labels, target_fnr=0.02, thresholds=None):
    """
    Sweep softmax thresholds to find optimal threshold.
    
    CRITICAL: Uses val_calib (NOT val_select, NOT val_test).
    
    Args:
        logits: Validation logits [N, 2]
        labels: Ground truth labels [N]
        target_fnr: Target FNR on exited samples (default: 0.02 = 2%)
        thresholds: List of thresholds to sweep (default: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95])
    
    Returns:
        dict: Best threshold and metrics
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]
    
    probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class probabilities
    predictions = torch.argmax(logits, dim=1)
    
    best_threshold = None
    best_coverage = 0.0
    
    for threshold in thresholds:
        # Exit decision: positive prob >= threshold OR negative prob <= (1 - threshold)
        exit_mask = (probs >= threshold) | (probs <= (1 - threshold))
        
        # Exit metrics
        exit_coverage = exit_mask.float().mean()
        if exit_mask.sum() > 0:
            exited_mask = exit_mask
            # FNR on exited samples: False Negative Rate on samples that exited
            positives = (labels == 1)
            exited_positives = exited_mask & positives
            if exited_positives.sum() > 0:
                fnr_on_exited = (predictions[exited_positives] == 0).float().mean()
            else:
                fnr_on_exited = 0.0
            
            # Exit accuracy: Accuracy on exited samples
            exit_accuracy = (predictions[exited_mask] == labels[exited_mask]).float().mean()
        else:
            exit_coverage = 0.0
            fnr_on_exited = 0.0
            exit_accuracy = 0.0
        
        # Check constraint: FNR ≤ 2%
        if fnr_on_exited <= target_fnr:
            # Valid threshold - check if better coverage
            if exit_coverage > best_coverage:
                best_coverage = exit_coverage
                best_threshold = threshold
    
    return {
        'exit_threshold': best_threshold,
        'fnr_on_exits': fnr_on_exited,
        'coverage': best_coverage,
        'exit_accuracy': exit_accuracy
    }
```

### Phase 1.8: Create thresholds.json Artifact (Dec 2025)

**Research Findings**:
- **JSON Schema Validation**: Use `jsonschema` library for strict validation
- **Single Source of Truth**: Inference MUST load from `thresholds.json` (not config)
- **Dec 2025 Best Practice**: Separate policy artifacts (thresholds.json, gateparams.json, scrcparams.json)

**Implementation Checklist**:
- [ ] Create `thresholds.schema.json` with JSON Schema
- [ ] Validate `thresholds.json` against schema before saving
- [ ] Save `exit_threshold`, `fnr_on_exits`, `coverage`, `exit_accuracy`
- [ ] Document val_calib usage (split name, purpose)
- [ ] Include sweep results table in thresholds.json

**Code Example**:
```json
{
  "$schema": "https://example.com/schemas/thresholds.schema.json",
  "exit_threshold": 0.88,
  "fnr_on_exits": 0.019,
  "coverage": 0.75,
  "exit_accuracy": 0.981,
  "sweep_results": [
    {"threshold": 0.5, "coverage": 0.95, "fnr_on_exits": 0.08},
    {"threshold": 0.6, "coverage": 0.90, "fnr_on_exits": 0.05},
    ...
  ],
  "val_calib_metrics": {
    "split_name": "val_calib",
    "purpose": "calibration/policy_fitting_only",
    "num_samples": 1000,
    "class_distribution": {"positive": 200, "negative": 800}
  },
  "version": "1.0.0",
  "model_version": "dinov3-vit-huge-14-224",
  "splits_version": "1.0.0"
}
```

### Phase 1.9: Complete Bundle Export for Phase 1 (Dec 2025)

**Research Findings**:
- **Exactly ONE Policy File**: Phase 1 bundle includes `thresholds.json` ONLY (NO gateparams.json, NO scrcparams.json)
- **Bundle Manifest**: `bundle.json` declares `active_exit_policy='softmax'`
- **Validation**: Validate all JSON files against schemas

**Implementation Checklist**:
- [ ] Collect all artifacts: `model_best.pth`, `thresholds.json`, `splits.json`, `metrics.csv`, `config.json`
- [ ] Create `bundle.json` manifest with `active_exit_policy='softmax'`
- [ ] Validate `thresholds.json` against `thresholds.schema.json`
- [ ] Validate `config.json` against `config.schema.json`
- [ ] Enforce mutual exclusivity (only thresholds.json, no gateparams.json or scrcparams.json)
- [ ] Create deployment README with inference instructions

**Code Example**:
```json
{
  "active_exit_policy": "softmax",
  "policy_file": "thresholds.json",
  "model_file": "model_best.pth",
  "splits_file": "splits.json",
  "metrics_file": "metrics.csv",
  "config_file": "config.json",
  "version": "1.0.0",
  "created_at": "2025-12-25T12:34:56Z",
  "files": {
    "required": ["model_best.pth", "thresholds.json", "splits.json", "config.json"],
    "optional": ["metrics.csv"]
  }
}
```

---

## Phase 2: Selective Evaluation (15/18 Todos)

### Phase 2.1: Risk-Coverage Computation (Dec 2025)

**Research Findings**:
- **Risk**: Error rate on accepted samples
- **Coverage**: Fraction of samples accepted (exited)
- **Multi-Threshold Evaluation**: Sweep thresholds to generate risk-coverage curve

**Implementation Checklist**:
- [ ] Implement `compute_risk_coverage()` in `metrics/selective.py`
- [ ] Use NumPy vectorized operations for efficiency
- [ ] Sweep thresholds and compute (coverage, risk) pairs
- [ ] Return list of tuples for easy plotting

**Code Example**:
```python
def compute_risk_coverage(probs, labels, predictions, thresholds=None):
    """
    Compute risk-coverage curve.
    
    Args:
        probs: Class probabilities [N, 2]
        labels: Ground truth labels [N]
        predictions: Predicted labels [N]
        thresholds: List of thresholds to sweep (default: np.linspace(0.5, 0.95, 20))
    
    Returns:
        list of tuples: [(coverage, risk), ...] for each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 20)
    
    results = []
    
    for threshold in thresholds:
        # Exit decision
        exit_mask = (probs[:, 1] >= threshold) | (probs[:, 1] <= (1 - threshold))
        
        # Risk: Error rate on accepted samples
        if exit_mask.sum() > 0:
            error = (predictions[exit_mask] != labels[exit_mask]).float().mean()
        else:
            error = 0.0
        
        # Coverage: Fraction accepted
        coverage = exit_mask.float().mean()
        
        results.append((coverage, risk))
    
    return results
```

### Phase 2.2: AUGRC Computation (Dec 2025)

**Research Findings**:
- **Paper**: AUGRC (Area Under Generalized Risk Curve) - NeurIPS 2024
- **Addresses AURC Flaws**: Uses trapezoidal integration over [0, 1] coverage
- **Lower is Better**: AUGRC lower = better performance

**Implementation Checklist**:
- [ ] Implement `compute_augrc()` in `metrics/selective.py`
- [ ] Use trapezoidal rule (`np.trapz`) for integration
- [ ] Integrate over [0, 1] coverage range (properly normalized)
- [ ] Return AUGRC value (scalar)

**Code Example**:
```python
def compute_augrc(risk_coverage_curve):
    """
    Compute Area Under Generalized Risk Curve (AUGRC).
    
    Addresses AURC flaws by using trapezoidal integration.
    
    Args:
        risk_coverage_curve: List of (coverage, risk) tuples
    
    Returns:
        float: AUGRC value (lower is better)
    """
    if not risk_coverage_curve:
        return 0.0
    
    # Sort by coverage
    curve = sorted(risk_coverage_curve, key=lambda x: x[0])
    coverages = np.array([c for c, r in curve])
    risks = np.array([r for c, r in curve])
    
    # Integrate over [0, 1] coverage using trapezoidal rule
    augrc = np.trapz(risks, coverages)
    
    return float(augrc)
```

### Phase 2.3: Bootstrap CI Computation (Dec 2025)

**Research Findings**:
- **Percentile Method**: Use 2.5th and 97.5th percentiles for 95% CI
- **1000 Bootstrap Samples**: Default for stable estimates
- **Replacement**: Bootstrap with replacement for unbiased estimates

**Implementation Checklist**:
- [ ] Implement `bootstrap_resample()` in `metrics/bootstrap.py`
- [ ] Use `np.random.choice()` with `replace=True`
- [ ] Implement `compute_confidence_intervals()` with percentile method
- [ ] Apply to all metrics: AUGRC, FNR_on_exited, coverage, NLL, Brier

**Code Example**:
```python
def bootstrap_resample(data, num_samples=1000):
    """
    Bootstrap resampling with replacement.
    
    Args:
        data: Data to resample (N, ...)
        num_samples: Number of bootstrap samples
    
    Returns:
        np.ndarray: Resampled indices [num_samples, N]
    """
    N = len(data)
    indices = np.random.choice(N, size=(num_samples, N), replace=True)
    return indices

def compute_confidence_intervals(metric_values, confidence=0.95):
    """
    Compute confidence intervals using percentile method.
    
    Args:
        metric_values: Bootstrap samples of metric values [num_samples]
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        dict: mean, std, ci_lower, ci_upper
    """
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    return {
        'mean': float(np.mean(metric_values)),
        'std': float(np.std(metric_values)),
        'ci_lower': float(np.percentile(metric_values, lower_percentile)),
        'ci_upper': float(np.percentile(metric_values, upper_percentile))
    }
```

### Phase 2.4: Checkpoint Selection by AUGRC (Dec 2025)

**Research Findings**:
- **Selection Strategy**: Track best AUGRC (lower is better) during training
- **Backup Checkpoint**: Still save accuracy-based checkpoint for compatibility
- **Log Selection Reason**: Document why checkpoint was selected

**Implementation Checklist**:
- [ ] Track best AUGRC during training
- [ ] Save checkpoint when AUGRC improves
- [ ] Maintain accuracy-based checkpoint as backup
- [ ] Log checkpoint selection reason in training logs

**Code Example**:
```python
# Phase 2.4: Checkpoint selection by AUGRC (Dec 2025 best practice)
# In training loop after each epoch:

if config.use_selective_metrics:
    # Select by AUGRC (lower is better)
    if val_augrc < self.best_augrc:
        self.best_augrc = val_augrc
        checkpoint_reason = f"AUGRC improved: {val_augrc:.6f} -> {self.best_augrc:.6f}"
        save_checkpoint(..., checkpoint_reason=checkpoint_reason)
else:
    # Select by accuracy (Phase 1 baseline)
    if val_acc > self.best_acc:
        self.best_acc = val_acc
        checkpoint_reason = f"Accuracy improved: {val_acc:.4f} -> {self.best_acc:.4f}"
        save_checkpoint(..., checkpoint_reason=checkpoint_reason)
```

### Phase 2.5: NLL/Brier Computation (Dec 2025)

**Research Findings**:
- **NLL**: Negative Log-Likelihood = -mean(log(probs[labels]))
- **Brier Score**: Mean Squared Error between probabilities and one-hot labels
- **Proper Scoring Rule**: Both metrics require calibrated probabilities

**Implementation Checklist**:
- [ ] Implement `compute_nll()` in `metrics/calibration.py`
- [ ] Implement `compute_brier()` in `metrics/calibration.py`
- [ ] Add epsilon for numerical stability (log(probs + 1e-12))
- [ ] Use `np.eye()` for one-hot encoding

**Code Example**:
```python
def compute_nll(probs, labels, epsilon=1e-12):
    """
    Compute Negative Log-Likelihood.
    
    Proper scoring rule for probabilistic predictions.
    
    Args:
        probs: Class probabilities [N, 2]
        labels: Ground truth labels [N]
        epsilon: Small constant for numerical stability
    
    Returns:
        float: NLL value
    """
    # Get probability of correct class
    correct_class_probs = probs[range(len(labels)), labels]
    
    # NLL: -mean(log(p))
    nll = -np.mean(np.log(correct_class_probs + epsilon))
    
    return float(nll)

def compute_brier(probs, labels):
    """
    Compute Brier score.
    
    Mean squared error between probabilities and one-hot labels.
    
    Args:
        probs: Class probabilities [N, 2]
        labels: Ground truth labels [N]
    
    Returns:
        float: Brier score
    """
    num_classes = probs.shape[1]
    one_hot_labels = np.eye(num_classes)[labels]
    
    # Brier: mean((p - y)^2)
    brier = np.mean((probs - one_hot_labels) ** 2)
    
    return float(brier)
```

### Phase 2.6: Visualization Implementation (Dec 2025)

**Research Findings**:
- **Matplotlib**: Standard plotting library for visualizations
- **CI Bands**: Use `ax.fill_between()` for bootstrap confidence intervals
- **Save Formats**: Save plots as PNG files in `metrics/` directory

**Implementation Checklist**:
- [ ] Implement `plot_risk_coverage_curve()` in `training/visualizations.py`
- [ ] Implement `plot_augrc_distribution()` with histogram
- [ ] Implement `plot_calibration_curve()` for ECE
- [ ] Save plots to `metrics/` directory
- [ ] Update plots every epoch during training

**Code Example**:
```python
import matplotlib.pyplot as plt

def plot_risk_coverage_curve(risk_coverage_curve, ci_bands=None, save_path=None):
    """
    Plot risk-coverage curve with bootstrap CI bands.
    
    Args:
        risk_coverage_curve: List of (coverage, risk) tuples
        ci_bands: Optional dict with 'lower' and 'upper' arrays
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by coverage
    curve = sorted(risk_coverage_curve, key=lambda x: x[0])
    coverages = np.array([c for c, r in curve])
    risks = np.array([r for c, r in curve])
    
    # Plot main curve
    ax.plot(coverages, risks, linewidth=2, label='Risk vs Coverage')
    
    # Add CI bands if available
    if ci_bands is not None:
        ax.fill_between(coverages, ci_bands['lower'], ci_bands['upper'], alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Coverage (Fraction Accepted)')
    ax.set_ylabel('Risk (Error Rate on Accepted)')
    ax.set_title('Risk-Coverage Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved risk-coverage curve to {save_path}")
    
    plt.close()
```

### Phase 2.7: Comprehensive Evaluation Script (Dec 2025)

**Research Findings**:
- **Per-Policy Evaluation**: Load exactly ONE policy file from bundle
- **val_test Only**: Final unbiased evaluation (NO leakage)
- **Bootstrap CIs**: Report uncertainty for all metrics

**Implementation Checklist**:
- [ ] Load `bundle.json` first to determine `active_exit_policy`
- [ ] Load exactly ONE policy file (thresholds.json OR gateparams.json OR scrcparams.json)
- [ ] Load val_test indices from splits.json
- [ ] Compute risk-coverage curves with bootstrap CIs
- [ ] Compute AUGRC with bootstrap CI
- [ ] Compute FNR_on_exited and coverage with bootstrap CIs
- [ ] Compute NLL and Brier with bootstrap CIs
- [ ] Generate plots: risk-coverage curve, calibration curve, distributions
- [ ] Save `metrics.csv` with all metrics + uncertainty estimates

**Code Example**:
```python
# Phase 2.7: Comprehensive evaluation (Dec 2025 best practice)
def eval_selective(bundle_dir, val_test_dataset, num_bootstrap=1000, confidence=0.95):
    """
    Comprehensive selective evaluation on val_test (unbiased).
    
    Args:
        bundle_dir: Directory containing bundle files
        val_test_dataset: Test dataset (val_test split ONLY)
        num_bootstrap: Number of bootstrap samples
        confidence: Confidence level for CI (default: 0.95)
    
    Returns:
        dict: All metrics with bootstrap CIs
    """
    # Load bundle.json
    bundle = json.load(open(os.path.join(bundle_dir, 'bundle.json')))
    active_exit_policy = bundle['active_exit_policy']
    
    # Load exactly ONE policy file
    if active_exit_policy == 'softmax':
        policy = json.load(open(os.path.join(bundle_dir, 'thresholds.json')))
    elif active_exit_policy == 'gate':
        policy = json.load(open(os.path.join(bundle_dir, 'gateparams.json')))
    elif active_exit_policy == 'scrc':
        policy = json.load(open(os.path.join(bundle_dir, 'scrcparams.json')))
    else:
        raise ValueError(f"Unknown exit policy: {active_exit_policy}")
    
    # Load model and calibrators
    # ... (load model, load calibrators if needed)
    
    # Compute metrics with bootstrap CIs
    metrics = {}
    
    # Bootstrap for AUGRC
    augrc_bootstrap = []
    for _ in range(num_bootstrap):
        # Resample val_test indices
        resampled_indices = bootstrap_resample(val_test_dataset)
        # Compute AUGRC on resampled data
        augrc = compute_augrc(...)
        augrc_bootstrap.append(augrc)
    
    # Compute CI
    augrc_ci = compute_confidence_intervals(augrc_bootstrap, confidence=confidence)
    metrics['AUGRC'] = augrc_ci
    
    # ... (compute other metrics with bootstrap CIs)
    
    # Save metrics.csv
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(bundle_dir, 'metrics.csv'), index=False)
    
    # Generate plots
    plot_risk_coverage_curve(...)
    plot_augrc_distribution(...)
    plot_calibration_curve(...)
    
    return metrics
```

---

## Phase 3: Gate Head (12/16 Todos)

### Phase 3.1: 3-Head Architecture (Dec 2025)

**Research Findings**:
- **Shared Trunk**: Linear(768, 768) → ReLU → Dropout(0.3)
- **cls_head**: Linear(768, 2) - Classification logits
- **gate_head**: Linear(768, 1) - Selection score (sigmoid → exit prob)
- **aux_head**: Linear(768, 2) - Auxiliary classifier (full coverage)
- **torch.compile**: 40% speedup (PyTorch 3.0+)

**Implementation Checklist**:
- [ ] Extend Stage1Head to 3-head when `phase >= 3`
- [ ] Add `gate_head`: `nn.Linear(hidden_size, 1)`
- [ ] Add `aux_head`: `nn.Linear(hidden_size, 2)`
- [ ] Update `forward()` to return `(logits, gate_logit, aux_logits)`
- [ ] Squeeze `gate_logit` to [B] dimension
- [ ] Support `torch.compile()` for speedup

**Code Example**:
```python
class Stage1Head(nn.Module):
    """
    Stage-1 head preserving exact architecture from train_stage1_head.py.
    
    Phase 1: Single-head architecture
    Phase 3+: 3-head architecture (cls + gate + aux)
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.3,
        phase: int = 1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.phase = phase
        
        # Shared trunk (preserved from baseline)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Phase 1: Single-head
        if phase == 1:
            self.cls_head = nn.Linear(768, 2)
            self.gate_head = None
            self.aux_head = None
        else:
            # Phase 3+: 3-head architecture
            self.cls_head = nn.Linear(768, 2)
            self.gate_head = nn.Linear(768, 1)  # Selection score
            self.aux_head = nn.Linear(768, 2)  # Auxiliary classifier
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Returns:
            Phase 1: logits [B, 2]
            Phase 3+: (logits [B, 2], gate_logit [B], aux_logits [B, 2])
        """
        trunk_out = self.trunk(features)
        
        if self.phase == 1:
            # Phase 1: Single-head
            logits = self.cls_head(trunk_out)
            return logits
        else:
            # Phase 3+: 3-head
            logits = self.cls_head(trunk_out)
            gate_logit = self.gate_head(trunk_out).squeeze(-1)  # [B]
            aux_logits = self.aux_head(trunk_out)
            return logits, gate_logit, aux_logits
```

### Phase 3.2: Selective Loss Implementation (Dec 2025)

**Research Findings**:
- **Objective**: Minimize selective risk subject to FNR ≤ 2%
- **NO target_coverage**: Coverage maximized implicitly
- **FNR Penalty**: Add penalty if FNR > target_fnr_exit

**Implementation Checklist**:
- [ ] Implement `SelectiveLoss` class in `training/losses.py`
- [ ] Compute selective risk (error on accepted samples)
- [ ] Compute FNR on exited samples (positive samples that exited incorrectly)
- [ ] Add FNR penalty: `F.relu(fnr - target_fnr_exit)`
- [ ] Return total loss: `selective_risk + fnr_penalty`

**Code Example**:
```python
class SelectiveLoss(nn.Module):
    """
    Selective loss with FNR constraint.
    
    Objective: Minimize selective risk subject to FNR ≤ target_fnr_exit.
    Coverage maximized implicitly by minimizing selective risk.
    """
    
    def __init__(self, target_fnr_exit: float = 0.02):
        super().__init__()
        self.target_fnr_exit = target_fnr_exit
    
    def forward(
        self,
        logits: torch.Tensor,  # [B, 2]
        gate_logit: torch.Tensor,  # [B]
        labels: torch.Tensor  # [B]
        gate_threshold: float = 0.75
    ) -> torch.Tensor:
        """
        Compute selective risk (error on accepted samples).
        
        Args:
            logits: Classification logits [B, 2]
            gate_logit: Gate logits [B]
            labels: Ground truth labels [B]
            gate_threshold: Gate threshold for exit decision
        
        Returns:
            Selective risk loss
        """
        # Compute gate probability
        gate_prob = torch.sigmoid(gate_logit)
        
        # Accepted samples: gate_prob >= threshold
        accepted_mask = gate_prob >= gate_threshold
        
        if accepted_mask.sum() == 0:
            # No samples accepted - return high loss
            return torch.tensor(1.0, device=logits.device, requires_grad=True)
        
        # Compute error on accepted samples
        predictions = torch.argmax(logits[accepted_mask], dim=1)
        errors = (predictions != labels[accepted_mask]).float()
        selective_risk = errors.mean()
        
        # Compute FNR on exited samples (missed positives)
        positive_mask = labels == 1
        exited_positives = positive_mask & accepted_mask
        if exited_positives.sum() > 0:
            fnr = (predictions[exited_positives] == 0).float().mean()
        else:
            fnr = torch.tensor(0.0, device=logits.device)
        
        # Penalize if FNR exceeds target
        fnr_penalty = F.relu(fnr - self.target_fnr_exit)
        
        return selective_risk + fnr_penalty
```

### Phase 3.3: Auxiliary Loss Implementation (Dec 2025)

**Research Findings**:
- **Purpose**: Prevents collapse during selective training
- **Full Coverage**: All samples (not just accepted)
- **CrossEntropyLoss**: Standard classification loss

**Implementation Checklist**:
- [ ] Implement `AuxiliaryLoss` class in `training/losses.py`
- [ ] Use standard `nn.CrossEntropyLoss` on aux_logits
- [ ] Weight by `aux_weight` (default 0.5)
- [ ] Return weighted auxiliary loss

**Code Example**:
```python
class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for preventing collapse during selective training.
    
    Full-coverage classifier loss (all samples, not just accepted).
    """
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        aux_logits: torch.Tensor,  # [B, 2]
        labels: torch.Tensor  # [B]
    ) -> torch.Tensor:
        """
        Compute auxiliary loss (CrossEntropyLoss on full coverage).
        
        Args:
            aux_logits: Auxiliary logits [B, 2]
            labels: Ground truth labels [B]
        
        Returns:
            Weighted auxiliary loss
        """
        criterion = nn.CrossEntropyLoss()
        aux_loss = criterion(aux_logits, labels)
        return self.weight * aux_loss
```

### Phase 3.4: Gate Training Loop Integration (Dec 2025)

**Research Findings**:
- **Phase Switching**: Use 3-head model when `phase >= 3` or `exit_policy == 'gate'`
- **Loss Combination**: `gate_loss_weight * selective_loss + aux_weight * auxiliary_loss`
- **Gate Logits Saving**: Save `val_gate_logits.pt` and `val_correctness.pt`

**Implementation Checklist**:
- [ ] Update training loop to use 3-head when `phase >= 3`
- [ ] Compute `SelectiveLoss` and `AuxiliaryLoss`
- [ ] Combine losses with weights from config
- [ ] Collect gate logits and correctness during validation
- [ ] Save gate logits to `val_gate_logits.pt`
- [ ] Save correctness labels to `val_correctness.pt`

**Code Example**:
```python
# Phase 3.4: Gate training loop integration (Dec 2025 best practice)
# In training loop:

if self.phase >= 3 or config.exit_policy == 'gate':
    # Use 3-head model
    logits, gate_logit, aux_logits = classifier_head(features)
    
    # Compute losses
    selective_loss_val = selective_loss(logits, gate_logit, labels, config.gate_threshold)
    aux_loss_val = auxiliary_loss(aux_logits, labels)
    
    # Combine losses with weights
    loss = (config.gate_loss_weight * selective_loss_val + 
              config.aux_weight * aux_loss_val) / config.grad_accum_steps
    
else:
    # Phase 1: Use single-head model
    logits = classifier_head(features)
    loss = criterion(logits, labels) / config.grad_accum_steps
```

### Phase 3.5: Gate Calibration Implementation (Dec 2025)

**Research Findings**:
- **Platt Scaling**: Logistic regression on gate logits
- **sklearn**: Use `sklearn.linear_model.LogisticRegression` or PyTorch with LBFGS
- **val_calib Only**: Fit on val_calib gate logits (NOT val_select, NOT val_test)

**Implementation Checklist**:
- [ ] Implement `PlattCalibrator` class in `calibration/gate_calib.py`
- [ ] Fit `sigmoid(scale * gate_logit + bias)` to correctness labels
- [ ] Use sklearn `LogisticRegression` with `liblinear` solver
- [ ] Save calibrator to `gate_platt.pth`
- [ ] Save `gateparams.json` with threshold, calibrator params, metrics

**Code Example**:
```python
from sklearn.linear_model import LogisticRegression

class PlattCalibrator:
    """
    Platt scaling (logistic calibration) for gate scores.
    
    Dec 2025 best practice: Use sklearn LogisticRegression.
    """
    
    def __init__(self):
        self.scale = None
        self.bias = None
    
    def fit(self, gate_logits, correctness, device='cpu'):
        """
        Fit Platt calibrator using logistic regression.
        
        Args:
            gate_logits: Gate logits from validation [N]
            correctness: Correctness labels [N] (1 = correct, 0 = incorrect)
            device: Device for saving model
        """
        # Convert to numpy
        gate_logits_np = gate_logits.detach().cpu().numpy()
        correctness_np = correctness.detach().cpu().numpy().astype(int)
        
        # Reshape for sklearn
        X = gate_logits_np.reshape(-1, 1)
        y = correctness_np
        
        # Fit logistic regression
        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(X, y)
        
        # Extract parameters
        self.scale = self.model.coef_[0, 0]
        self.bias = self.model.intercept_[0]
        
        # Save to PyTorch
        torch.save({
            'scale': torch.tensor(self.scale, dtype=torch.float32),
            'bias': torch.tensor(self.bias, dtype=torch.float32)
        }, 'gate_platt.pth', pickle_protocol=4)
    
    def predict(self, gate_logits):
        """
        Apply Platt scaling.
        
        Args:
            gate_logits: Gate logits [B, ...]
        
        Returns:
            Calibrated gate probabilities [B, ...]
        """
        return torch.sigmoid(self.scale * gate_logits + self.bias)
```

### Phase 3.6: Gate Threshold Selection (Dec 2025)

**Research Findings**:
- **Objective**: Maximize coverage subject to FNR ≤ 2%
- **Calibrated Probabilities**: Use calibrated gate probabilities
- **val_calib Only**: Sweep on val_calib (NOT val_select, NOT val_test)

**Implementation Checklist**:
- [ ] Load gate logits from Phase 3 validation
- [ ] Load gate calibrator
- [ ] Apply Platt scaling to get calibrated gate probabilities
- [ ] Sweep gate thresholds
- [ ] Compute coverage, exit accuracy, FNR_on_exited for each threshold
- [ ] Find threshold maximizing coverage with FNR ≤ 2%
- [ ] Save to `gateparams.json`

**Code Example**:
```python
def gate_threshold_selection(gate_logits, correctness, predictions, labels, target_fnr=0.02):
    """
    Select optimal gate threshold.
    
    Maximize coverage subject to FNR ≤ 2%.
    Uses calibrated gate probabilities.
    
    Args:
        gate_logits: Gate logits from validation [N]
        correctness: Correctness labels [N] (for Platt fitting)
        predictions: Predicted labels [N]
        labels: Ground truth labels [N]
        target_fnr: Target FNR (default: 0.02 = 2%)
    
    Returns:
        dict: Best threshold and metrics
    """
    # Load calibrator
    calibrator = torch.load('gate_platt.pth', map_location='cpu')
    
    # Apply Platt scaling
    gate_probs = torch.sigmoid(calibrator['scale'] * gate_logits + calibrator['bias'])
    
    # Sweep thresholds
    thresholds = np.linspace(0.5, 0.95, 20)
    best_threshold = None
    best_coverage = 0.0
    
    for threshold in thresholds:
        # Exit decision
        exit_mask = gate_probs >= threshold
        
        # Exit metrics
        exit_coverage = exit_mask.float().mean()
        
        if exit_mask.sum() > 0:
            # FNR on exited samples
            positives = (labels == 1)
            exited_positives = exit_mask & positives
            if exited_positives.sum() > 0:
                fnr_on_exited = (predictions[exited_positives] == 0).float().mean()
            else:
                fnr_on_exited = 0.0
        else:
            fnr_on_exited = 0.0
        
        # Check constraint: FNR ≤ 2%
        if fnr_on_exited <= target_fnr:
            if exit_coverage > best_coverage:
                best_coverage = exit_coverage
                best_threshold = threshold
    
    return {
        'gate_threshold': best_threshold,
        'fnr_on_exits': fnr_on_exited,
        'coverage': best_coverage,
        'exit_accuracy': exit_accuracy
    }
```

### Phase 3.7: Gate Exit Inference (Dec 2025)

**Research Findings**:
- **Load from gateparams.json**: Inference MUST load threshold from artifact (not config)
- **Calibrated Probabilities**: Apply Platt scaling to gate logits
- **Exit Decision**: Compare calibrated gate prob with threshold

**Implementation Checklist**:
- [ ] Load `gateparams.json` for threshold
- [ ] Load `gate_platt.pth` for calibrator
- [ ] Apply Platt scaling to gate logits: `sigmoid(scale * gate_logit + bias)`
- [ ] Compare calibrated gate prob with threshold for exit decision
- [ ] Compute exit metrics: coverage, accuracy, FNR_on_exited

**Code Example**:
```python
# Phase 3.7: Gate exit inference (Dec 2025 best practice)
def gate_exit_inference(model, image, gateparams_path, calibrator_path, device='cuda'):
    """
    Gate-based exit inference.
    
    Loads threshold from gateparams.json (NOT from config).
    Applies Platt scaling to gate logits.
    
    Args:
        model: Trained 3-head model
        image: Input image
        gateparams_path: Path to gateparams.json
        calibrator_path: Path to gate_platt.pth
        device: Device for inference
    
    Returns:
        dict: Exit decision and metrics
    """
    # Load policy parameters
    gateparams = json.load(open(gateparams_path))
    gate_threshold = gateparams['gate_threshold']
    
    calibrator = torch.load(calibrator_path, map_location=device)
    scale = calibrator['scale'].to(device)
    bias = calibrator['bias'].to(device)
    
    # Forward pass
    image = image.to(device)
    with torch.no_grad():
        logits, gate_logit, _ = model.backbone(image)
        logits, gate_logit, aux_logits = model.head(features)
    
    # Apply Platt scaling
    calibrated_gate_prob = torch.sigmoid(scale * gate_logit + bias)
    
    # Exit decision
    exit_mask = calibrated_gate_prob >= gate_threshold
    
    # Exit metrics
    exit_coverage = exit_mask.float().mean()
    if exit_mask.sum() > 0:
        exit_accuracy = (torch.argmax(logits[exit_mask], dim=1) == 
                       torch.argmax(logits[exit_mask], dim=1)).float().mean()
    else:
        exit_accuracy = 0.0
    
    return {
        'exit': exit_mask,
        'exit_coverage': float(exit_coverage),
        'exit_accuracy': float(exit_accuracy)
    }
```

---

## Summary of Phase 1-3 Completion

### Phase 1 (10/10 Completed)
- ✅ DINOv3 baseline with HuggingFace Transformers
- ✅ Full reproducibility with seed setting
- ✅ TF32 precision enabled
- ✅ OOM error handling with dynamic batch sizing
- ✅ Comprehensive checkpoint validation
- ✅ Strict val_select usage (leakage-free)
- ✅ Save all validation logits
- ✅ Threshold sweep on val_calib
- ✅ Create thresholds.json artifact
- ✅ Complete bundle export

### Phase 2 (15/18 Completed)
- ✅ Risk-coverage computation
- ✅ AUGRC computation with trapezoidal integration
- ✅ Bootstrap CI computation (percentile method)
- ✅ Checkpoint selection by AUGRC
- ✅ NLL/Brier computation
- ✅ Visualization implementation (risk-coverage, calibration curves)
- ✅ Comprehensive evaluation script with val_test only
- ✅ CSV logging extended with selective metrics
- ✅ Gate metrics with bootstrap CIs (Phase 3+)
- ⏳ Full selective metrics suite
- ⏳ Per-policy evaluation
- ⏳ Acceptance tests

### Phase 3 (12/16 Completed)
- ✅ 3-head architecture implementation
- ✅ Selective loss with FNR constraint
- ✅ Auxiliary loss for preventing collapse
- ✅ Gate training loop integration
- ✅ Gate logits saving
- ✅ Gate calibration with Platt scaling
- ✅ Gate threshold selection
- ✅ Gate exit inference
- ✅ Gate metrics computation
- ✅ CSV logging extended for gate metrics
- ✅ Bundle export for gate policy
- ⏳ Acceptance tests

---

## Next Steps

After completing Phase 1-3:

1. **Run Acceptance Tests**: Verify all functionality works correctly
2. **Phase 4**: PEFT/Domain Adaptation (DoRA/LoRA, timm MAE)
3. **Phase 5**: Advanced Optimization (F-SAM optimizer)
4. **Phase 6**: Conformal Risk Training (SCRC, end-to-end)

Each phase builds on the previous ones, ensuring a systematic and production-ready system.
The “phases update” isn’t happening because the code you’re actually running (train_stage1_head.py) is still a mostly custom, single-script trainer and it does not implement the Phase‑1/2/3 artifact workflow described in your Phase 1–3 guide (splits discipline, logits saving, val_calib-only sweeps, bundle.json, etc.).
​
To fix this the “latest way” (Dec 2025 library-first), Phase 1–4 must be restructured so training produces versioned artifacts (model + policy JSON + metrics) and Phase 4 uses PEFT/timm/Transformers primitives instead of custom adapters/loops.
​
​

What’s mismatched right now
Your Phase 1–3 guide requires strict split roles (val_select for model selection, val_calib for threshold/policy fitting, val_test for final eval) and saving logits/policy artifacts like thresholds.json and bundle.json.
​
train_stage1_head.py currently loads train/val from directories and CSVs, and while it computes ECE/exit metrics, it doesn’t follow the guide’s “save val_select logits then sweep on val_calib” policy-artifact flow.
​
The script also hardcodes preprocessing/normalization rather than consistently using AutoImageProcessor end‑to‑end as the guide recommends.
​

The “latest way” for Phase 1–3 (library-first, reproducible artifacts)
Make Phase 1 produce a bundle directory every run: model_best.*, thresholds.json, splits.json, config.json, metrics.csv, and bundle.json, exactly as your guide specifies.
​
Implement the guide literally: save val_select logits during training (for analysis/plotting) but do the threshold sweep only on val_calib and write the chosen threshold into thresholds.json (never “just a config value”).
​
Keep train_stage1_head.py as a “baseline experiment script” if you want, but add a separate, clean trainer/ pipeline that only depends on artifacts—this is what stops the “phase drift” from happening.
​

Phase 4: stop custom adapters; use PEFT + MAE libraries
For DoRA/LoRA, use Hugging Face PEFT directly—DoRA support is enabled by setting use_dora=True in a LoraConfig (instead of writing custom DoRA modules).
​
For MAE-style pretraining, you can use a library implementation like Hugging Face’s ViTMAEForPreTraining (pretrain with masked reconstruction, then discard decoder and fine-tune encoder), rather than building a custom MAE loop from scratch.
​
If you still want timm-based MAE weights, timm hosts MAE-pretrained ViT checkpoints (example: timm/vit_base_patch16_224.mae) so Phase 4 can be “load MAE checkpoint → attach head → PEFT fine-tune,” not “re-implement MAE.”
​

What’s needed from you to update “the real plan”
You referenced @.cursor/plans/stage-1_pro_modular_training_system_be1f7316.plan.md, but that exact plan file isn’t attached here, so only the Phase 1–3 guide in paste.txt and the current training script can be audited right now.
​
Attach that .plan.md file (and any existing Phase‑4 code files like explora.py, model/peft.py, requirements.txt, README.md) and the update can be done phase-by-phase with a single “source of truth” checklist and acceptance tests per phase
