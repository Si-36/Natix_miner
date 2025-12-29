# Complete Train CLI Refactor - 2025-12-29 Elite Production

## What's Being Fixed

### **CRITICAL FIXES**:
1. ✅ **Zero Hardcoding**: All paths/hyperparams from `cfg`
2. ✅ **Phase 1 → Phase 4 Integration**: Load ExPLoRA checkpoint if `cfg.model.init_from_explora=true`
3. ✅ **Phase 2 Implementation**: Threshold sweep (CPU-fast, required for export)
4. ✅ **Phase 5 Implementation**: Temperature scaling calibration
5. ✅ **Phase 6 Implementation**: Bundle export (deployable artifact)
6. ✅ **Metric Logging Fix**: Log `val_select/acc` (not just `val/acc`) for early stopping
7. ✅ **Phase 1 Writes metrics.csv**: Required by contract
8. ✅ **ValCalibArtifactSaver Optimization**: Reuse `validation_step` outputs (no extra forward pass)

---

## Config → Code Mapping

### **Phase 1 Executor Reads**:
```python
# From cfg.data
DATA_ROOT = cfg.data.data_root  # "/data/natix"
BATCH_SIZE = cfg.data.dataloader.batch_size  # 32
NUM_WORKERS = cfg.data.dataloader.num_workers  # 8

# From cfg.model
BACKBONE_ID = cfg.model.backbone_id  # "facebook/dinov3-vith16-pretrain-lvd1689m"
NUM_CLASSES = cfg.model.num_classes  # 13
FREEZE_BACKBONE = cfg.model.freeze_backbone  # true
HEAD_TYPE = cfg.model.head_type  # "linear"
DROPOUT_RATE = cfg.model.dropout_rate  # 0.3
USE_EMA = cfg.model.use_ema  # true
USE_MULTIVIEW = cfg.model.use_multiview  # true
INIT_FROM_EXPLORA = cfg.model.init_from_explora  # false (set true to use Phase 4)

# From cfg.training
MAX_EPOCHS = cfg.training.epochs  # 100
LEARNING_RATE = cfg.training.optimizer.lr  # 1e-4
WEIGHT_DECAY = cfg.training.optimizer.weight_decay  # 0.01

# From cfg.hardware
NUM_GPUS = cfg.hardware.num_gpus  # 1
PRECISION = "16-mixed" if cfg.training.mixed_precision.enabled else "32"

# From cfg.early_stopping (CRITICAL FIX)
MONITOR_METRIC = cfg.training.early_stopping.monitor  # "val_select/acc"
MONITOR_MODE = cfg.training.early_stopping.mode  # "max"
PATIENCE = cfg.training.early_stopping.patience  # 10
```

---

## Phase 1: ExPLoRA Integration Logic

```python
def phase1_executor(artifacts, cfg):
    # ... create datamodule, model ...

    # CRITICAL FIX: Load ExPLoRA checkpoint if requested
    if cfg.model.init_from_explora and artifacts.explora_checkpoint.exists():
        logger.info(f"Loading ExPLoRA checkpoint: {artifacts.explora_checkpoint}")
        explora_state = torch.load(artifacts.explora_checkpoint, map_location="cpu")
        model.net["backbone"].model.load_state_dict(explora_state, strict=False)
        logger.info("✅ Loaded ExPLoRA-adapted backbone (Phase 4 → Phase 1)")
    elif cfg.model.init_from_explora:
        logger.warning(f"ExPLoRA requested but checkpoint not found: {artifacts.explora_checkpoint}")

    # ... training ...

    # CRITICAL FIX: Write metrics.csv (required by contract)
    metrics_df = pd.DataFrame(trainer.logger.metrics)
    metrics_df.to_csv(artifacts.metrics_csv, index=False)
    logger.info(f"Saved metrics: {artifacts.metrics_csv}")
```

---

## Phase 2: Threshold Sweep (Simple, Fast)

```python
def phase2_executor(artifacts, cfg):
    """
    Phase 2: Threshold Sweep

    Contract:
    - Input: val_calib_logits.pt, val_calib_labels.pt
    - Output: thresholds.json
    - Allowed split: VAL_CALIB only
    """
    import torch
    import json
    from sklearn.metrics import f1_score

    # Load val_calib logits/labels
    logits = torch.load(artifacts.val_calib_logits)
    labels = torch.load(artifacts.val_calib_labels)

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Sweep thresholds (0.1 to 0.9, step 0.05)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 0.95, 0.05):
        preds = (probs.max(dim=-1).values > threshold).long()
        f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Save to thresholds.json
    thresholds = {
        "threshold": float(best_threshold),
        "f1_score": float(best_f1),
        "method": "global_threshold",
    }

    with open(artifacts.thresholds_json, "w") as f:
        json.dump(thresholds, f, indent=2)

    logger.info(f"✅ Phase 2 complete: threshold={best_threshold:.3f}, F1={best_f1:.3f}")
```

---

## Phase 5: Temperature Scaling (SCRC)

```python
def phase5_executor(artifacts, cfg):
    """
    Phase 5: SCRC Calibration (Temperature Scaling)

    Contract:
    - Input: val_calib_logits.pt, val_calib_labels.pt
    - Output: scrcparams.json
    - Allowed split: VAL_CALIB only
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Load val_calib logits/labels
    logits = torch.load(artifacts.val_calib_logits)
    labels = torch.load(artifacts.val_calib_labels)

    # Temperature scaling: calibrate temperature parameter
    temperature = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = nn.functional.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    # Save calibration params
    params = {
        "method": "temperature_scaling",
        "temperature": float(temperature.item()),
        "calibration_loss": float(eval().item()),
    }

    with open(artifacts.scrcparams_json, "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"✅ Phase 5 complete: temperature={temperature.item():.4f}")
```

---

## Phase 6: Bundle Export

```python
def phase6_executor(artifacts, cfg):
    """
    Phase 6: Bundle Export

    Contract:
    - Input: phase1_checkpoint, splits.json
    - Output: bundle.json
    - Packages: model + ONE policy (threshold OR scrc)
    """
    import json
    import shutil

    # Determine which policy to use
    if artifacts.scrcparams_json.exists():
        policy_path = artifacts.scrcparams_json
        policy_type = "scrc"
    elif artifacts.thresholds_json.exists():
        policy_path = artifacts.thresholds_json
        policy_type = "threshold"
    else:
        raise FileNotFoundError("No policy file found (need thresholds.json or scrcparams.json)")

    # Create bundle manifest
    bundle = {
        "model_checkpoint": str(artifacts.phase1_checkpoint.relative_to(artifacts.output_dir)),
        "policy_type": policy_type,
        "policy_path": str(policy_path.relative_to(artifacts.output_dir)),
        "splits_json": str(artifacts.splits_json.relative_to(artifacts.output_dir)),
        "num_classes": cfg.model.num_classes,
        "backbone_id": cfg.model.backbone_id,
        "created_at": pd.Timestamp.now().isoformat(),
    }

    # Write bundle.json
    with open(artifacts.bundle_json, "w") as f:
        json.dump(bundle, f, indent=2)

    logger.info(f"✅ Phase 6 complete: bundle exported to {artifacts.bundle_json}")
    logger.info(f"   Model: {artifacts.phase1_checkpoint}")
    logger.info(f"   Policy: {policy_path} ({policy_type})")
```

---

## Next Steps

1. ✅ **Confirm approach** - Does this look correct?
2. ⏭️ **Implement refactored train_cli.py** with all phases
3. ⏭️ **Fix module.py** to log `val_select/acc`
4. ⏭️ **Optimize callback** to reuse validation outputs
5. ⏭️ **Test full pipeline** end-to-end

**Estimated LOC**: ~800 lines for complete train_cli.py
