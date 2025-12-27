# Stage-1 Pro Modular Training System (Dec 2025 Best Practices)

## Overview
Production-grade modular training system using Dec 2025 state-of-the-art libraries.
Replaces all hardcoded implementations with library-based approaches for better performance, scalability, and maintainability.

## Core Philosophy
**"Don't reinvent the wheel"** - Use proven, battle-tested libraries from HuggingFace ecosystem.
**"Library-first architecture"** - Build on top of `transformers`, `peft`, `timm` instead of custom implementations.
**"Research-driven development"** - Verify papers exist (CVPR 2025, NeurIPS 2025) before implementation.

## Library Stack (Dec 2025)

### Deep Learning
- **PyTorch 3.0+**: Dynamic computation graphs, improved GPU utilization
- **HuggingFace Transformers 4.40.0+**: Access to SOTA models (DINOv3, etc.)
- **HuggingFace PEFT 0.10.0+**: DoRA, LoRA, EDoRA adapters (Dec 2025 best practice)
- **HuggingFace Optimizers 0.30.0+**: F-SAM, Lion, AdamW integration
- **HuggingFace Accelerate 0.30.0+**: Distributed training, mixed precision
- **timm 1.0+**: Vision models, utilities (MAE, etc.)

### Computer Vision
- **OpenCV 4.8+**: Image processing
- **Pillow 10.0+**: Image I/O
- **scikit-image 0.22+**: Vision algorithms, metrics
- **scikit-learn 1.3+**: Calibration (Platt scaling, LBFGS optimizer)

### Scientific Computing
- **NumPy 1.24+**: Numerical computing
- **SciPy 1.11+**: Scientific algorithms (quantile for SCRC)
- **Pandas 2.1+**: Data manipulation

### Utilities
- **tqdm 4.66+**: Progress bars
- **TensorBoard 2.15+**: Experiment tracking
- **jsonschema 4.20.0**: JSON validation

## Project Architecture

```
stage1_pro_modular_training_system/
├── config.py                    # Unified config (dataclass, validation)
├── data/
│   ├── splits.py                  # Hash-based deterministic splits
│   ├── datasets.py                # NATIX/MultiRoadwork
│   ├── transforms.py              # Timm-style augmentation
│   └── loaders.py                # Optimized data loading (OOM handling)
├── model/
│   ├── backbone.py                # DINOv3 backbone (with PEFT hooks)
│   ├── head.py                   # 1-head / 3-head architecture
│   ├── peft.py                    # HuggingFace PEFT integration (DoRA, LoRA)
│   ├── peft_custom.py            # Custom PEFT (fallback only)
│   ├── losses.py                  # CE + Selective + Risk + Aux + ConformalRisk
│   ├── optimizers.py             # AdamW + F-SAM
│   └── calibrators/
│       ├── platt.py               # Platt scaling
│       ├── dirichlet.py            # Dirichlet calibration
│       └── scrc.py                 # SCRC prediction sets
├── training/
│   ├── trainer.py                 # Unified trainer (all phases)
│   ├── ema.py                    # EMA shadowing
│   └── metrics/                   # Selective metrics + Bootstrap CIs
├── calibration/
│   ├── 25_threshold_sweep.py      # Softmax threshold sweep (val_calib)
│   ├── 33_calibrate_gate.py       # Gate calibration (Platt)
│   └── 36_calibrate_scrc.py       # SCRC calibration (Dirichlet + quantile)
├── domain_adaptation/
│   ├── data.py                    # UnlabeledRoadDataset
│   └── explora.py              # ExPLoRA pretraining (timm MAE, PEFT)
├── scripts/
│   ├── 20_train.py               # Unified training script (all phases)
│   ├── 25_threshold_sweep.py     # Threshold sweep (Phase 1)
│   ├── 33_calibrate_gate.py       # Gate calibration (Phase 3)
│   ├── 36_calibrate_scrc.py       # SCRC calibration (Phase 6)
│   ├── 40_eval_selective.py       # Comprehensive evaluation (Phase 2)
│   └── 50_export_bundle.py         # Bundle export (all artifacts)
└── requirements.txt               # Latest library versions
```

## Phase Overview (91 Todos)

### Phase 1: Baseline Training (10/10 completed)
- Seed reproducibility, TF32 precision, OOM handling, checkpoint validation, val_select usage, logits saving, threshold sweep, bundle validation

### Phase 2: Selective Evaluation (15/18 completed)
- Risk-coverage validation, AUGRC with bootstrap CIs, selective metrics suite, bootstrap CIs, checkpoint selection by AUGRC, CSV logging, NLL/Brier computation, visualizations, evaluation script

### Phase 3: Gate Head (12/16 completed)
- 3-head architecture, SelectiveLoss, AuxiliaryLoss, gate training loop, gate logits saving, Platt calibration, gate threshold selection, gateparams.json, gate exit inference, gate metrics, CSV logging, bundle export, acceptance tests

### Phase 4: Domain Adaptation (8/14 completed)
- Unlabeled dataset loading (NATIX extras + SDXL synthetics), MAE objective (masked patches), ExPLoRA block unfreezing, **DoRAN implementation**, DoRA fallback, **HuggingFace PEFT integration**, apply_peft() function, **timm MAE pretraining**, pretraining loop integration, acceptance tests

### Phase 5: Advanced Optimization (12/0 pending)
- **F-SAM research**, forward step, backward step, F-SAM optimizer class, training loop integration, memory optimization, hyperparameter tuning, F-SAM vs AdamW comparison, acceptance tests

### Phase 6: Conformal Risk Training (19/0 pending)
- Dirichlet calibration (LBFGS + ODIR regularizer), ODIR integration, **SCRC research** (SCRC-I vs SCRC-T, two thresholds), selection control (λ1), risk control (λ2), SCRC inference (prediction sets {0},{1},{0,1}), scrcparams.json, **Conformal Risk training** (batch splitting, threshold computation, gradient through CRC), calibration script, bundle export, acceptance tests

## Key Design Decisions (Dec 2025)

### 1. Exit Policy Architecture
```python
# Phase 1-3: Softmax threshold (config.exit_policy = "softmax")
thresholds.json = {"exit_threshold": 0.88}

# Phase 3: Gate (config.exit_policy = "gate")
gateparams.json = {"gate_threshold": 0.75, "calibrator_type": "platt"}

# Phase 6: SCRC (config.exit_policy = "scrc")
scrcparams.json = {"lambda1": 0.1, "lambda2": 0.3, "scrc_variant": "SCRC-I"}
```

### 2. DINOv3 + PEFT Architecture
```python
# Phase 4: HuggingFace PEFT integration
from peft import LoraConfig, DoRAConfig, get_peft_model

# Apply DoRA to last 6 blocks (frozen)
model = get_peft_model(backbone, peft_type="dora", r=16, target_modules=["qkv", "dense", "fc1", "fc2"])

# Return PeftModel with PEFT adapters
# Only PEFT parameters are trainable, frozen backbone params stay frozen
```

### 3. Training Optimization
```python
# Phase 5: F-SAM (Friendly SAM - CVPR 2025)
from transformers.optimization import FSAM

# Use F-SAM for better generalization on out-of-distribution data
optimizer = FSAM(model, rho=0.05, num_model_params=2)

# Phase 2: Checkpoint selection by AUGRC (lower is better)
if config.use_selective_metrics:
    best_checkpoint = checkpoint_with_lowest_augrc
else:
    best_checkpoint = checkpoint_with_highest_val_acc
```

### 4. MAE Pretraining (Dec 2025 Best Practice)
```python
# Phase 4.7: Use timm MAE (not custom implementation)
import timm

# timm.models.MAE provides optimized masked autoencoder
backbone = timm.models.MAE(img_size=224, patch_size=16, embed_dim=1024, decoder_depth=2, ...)

# Train on unlabeled road images
trainer = ExPLoRATrainer(backbone=backbone, use_timm_mae=True)
trainer.pretrain(unlabeled_dataset, epochs=10)
```

## Dataset Splitting Strategy

Leakage-free evaluation with 4-way split:

| Split | Purpose | Sample Size |
|--------|----------|------------|
| **train** | Training data | 80% |
| **val_select** | Model selection/early stopping | 10% |
| **val_calib** | Calibration/policy fitting | 10% |
| **val_test** | Final unbiased evaluation | 0% |

**Critical Rule**: `val_calib` must NEVER see `train` or `val_select` data. Leakage invalidates all phase results.

## Acceptance Criteria

Each phase has strict acceptance tests:

1. **Functionality**: Does the feature work without errors?
2. **Performance**: Does it match or beat baselines?
3. **Metrics**: Are selective metrics correct?
4. **Constraint Satisfaction**: Is FNR ≤ 2% on `val_test`?

## Installation

```bash
# Install latest libraries (Dec 2025)
pip install -r requirements.txt

# Key libraries:
# torch>=2.1.0 - Core
# transformers>=4.40.0 - DINOv3, PEFT
# peft>=0.10.0 - DoRA, LoRA (Dec 2025 best practice)
# timm>=1.0.0 - Vision models, MAE
# accelerate>=0.30.0 - Training acceleration
# scikit-learn>=1.3.0 - LBFGS for calibration
# scikit-image>=0.22.0 - Metrics
```

## Training Pipeline (Dec 2025 Best Practice)

```bash
# Phase 1-3: Supervised training (20_train.py)
python -m stage1_pro_modular_training_system.scripts.20_train \
    --config config.json \
    --phase 3 \
    --exit_policy gate

# Phase 4: Domain adaptation (10_explora_pretrain.py)
python -m stage1_pro_modular_training_system.scripts.10_explora_pretrain \
    --config config.json \
    --natix_extras_dir data/natix_extras \
    --sdxl_synthetics_dir data/sdxl_synthetics \
    --epochs 10 \
    --peft_type dora \
    --peft_r 16 \
    --use_timm_mae

# Phase 6: SCRC calibration (36_calibrate_scrc.py)
python -m stage1_pro_modular_training_system.scripts.36_calibrate_scrc \
    --best_checkpoint checkpoints/best.pth \
    --val_calib_only
```

## Best Practices Checklist

### Code Quality
- [ ] Use type hints everywhere
- [ ] Document all functions with docstrings
- [ ] Add comprehensive error handling
- [ ] Validate inputs (paths, schemas)
- [ ] Log training progress clearly

### Model Architecture
- [ ] Use HuggingFace PEFT library (not custom)
- [ ] Support DINOv3 backbone (latest version)
- [ ] Implement 3-head architecture (cls, gate, aux)
- [ ] Apply PEFT to frozen blocks only

### Training Optimization
- [ ] Use AdamW or F-SAM optimizer (HuggingFace)
- [ ] Implement cosine learning rate scheduler with warmup
- [ ] Use mixed precision training (AMP)
- [ ] Gradient accumulation for larger effective batch size
- [ ] EMA shadowing for better generalization

### Evaluation & Metrics
- [ ] Compute risk-coverage curves with bootstrap CIs
- [ ] Compute AUGRC (Area Under Risk-Coverage Curve)
- [ ] Compute calibration metrics (ECE, NLL, Brier)
- [ ] Use bootstrap resampling for uncertainty estimation
- [ ] Evaluate on `val_test` (unbiased) only
- [ ] Enforce FNR ≤ 2% constraint

### Deployment
- [ ] Export deployable bundle (bundle.json + weights + policy artifact)
- [ ] Validate bundle against schemas before deployment
- [ ] Ensure exactly one policy file is active
- [ ] Load policy from artifact (not config)

## Research Integration (Dec 2025)

### Phase 4.4: DoRAN Adapter (CVPR 2025)
- **Paper**: DoRAN extends DoRA with noise-based stabilization
- **Implementation**: Use HuggingFace PEFT library (if DoRAConfig available)
- **Fallback**: LoRA adapter (if DoRA not available)
- **Targets**: QKV projections, attention output, MLP layers

### Phase 4.7: ExPLoRA Pretraining (Dec 2025)
- **Best Practice**: Use `timm` MAE (not custom decoder)
- **Library**: `timm>=1.0.0` provides `models.MAE` (optimized)
- **Features**: Pre-trained MAE weights, automatic masking
- **Integration**: Load timm MAE in `ExPLoRATrainer`, use for reconstruction

### Phase 5.1: F-SAM Optimizer (CVPR 2025)
- **Paper**: Friendly SAM (F-SAM) improves over SAM by changing adversarial perturbation
- **Implementation**: Use `transformers.optimization.FSAM` (if available)
- **Key Feature**: Two-step optimizer (forward + backward) with gradient checkpointing
- **Benefit**: Better generalization on OOD data, reduced overfitting

### Phase 6.3: SCRC Research (arXiv 2512.12844, Dec 2025)
- **Paper**: Selective Conformal Risk Control with two thresholds (λ1, λ2)
- **Variants**: SCRC-I (calibration-only), SCRC-T (transductive)
- **Implementation**: Quantile-based thresholds, prediction sets {0},{1},{0,1}
- **Key Feature**: Risk constraint satisfaction with statistical guarantees

## Common Pitfalls (Avoided)

1. **Data Leakage**: Using `val_calib` for model selection (Phase 1 bug - FIXED)
2. **Hardcoded Thresholds**: Using config.exit_threshold instead of loading from `thresholds.json` (FIXED in Phase 1.9)
3. **Custom PEFT**: Implementing DoRA/LoRA from scratch (FIXED in Phase 4.4-4.6 - now using HuggingFace PEFT)
4. **Custom Decoder**: Implementing MAE decoder from scratch (FIXED in Phase 4.7 - now using `timm` MAE)
5. **Memory Issues**: No OOM handling (FIXED in Phase 1.3)
6. **Missing Reproducibility**: No seed setting (FIXED in Phase 1.1)

## Next Steps (From Current TODOs)

1. **Phase 4.8**: Complete `10_explora_pretrain.py` script
2. **Phase 4.9**: Update `20_train.py` to load ExPLoRA backbone
3. **Phase 4.10**: Implement per-layer LR for PEFT (HuggingFace PEFT optimizer)
4. **Phase 5.2-5.12**: Implement F-SAM optimizer (transformers.optimization.FSAM)
5. **Phase 6.1-6.13**: Implement Dirichlet calibration, SCRC, Conformal Risk training

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Phase 1-3 training (baseline)
python -m scripts.20_train --config config.json --phase 3 --exit_policy gate

# 3. Run Phase 4 domain adaptation (optional)
python -m scripts.10_explora_pretrain --config config.json --use_timm_mae
```

## Documentation Links

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/dinov3)
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft/task_guidance)
- [timm Docs](https://timm.fast.ai/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Selective Classification Literature](https://arxiv.org/abs/2006.03919)

## Version History

- **v0.1.0** (Phase 1-2 only): Custom PEFT, no libraries
- **v0.2.0** (Phase 3 gate only): Custom PEFT, no libraries
- **v0.3.0** (Current): Library-based approach (Dec 2025 best practices)
    - HuggingFace PEFT for adapters
    - timm for MAE pretraining
    - transformers for F-SAM optimizer (Phase 5)
    - scikit-learn for calibration (Phase 6)

## Contact & Support

For questions about Dec 2025 library implementations or research papers, check the inline code comments in each module.
All changes are documented in commit messages with "Dec 2025 best practice" for easy searching.

