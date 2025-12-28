# ULTIMATE COMPLETE 210-TODO IMPLEMENTATION PLAN
## Production-Grade Multi-View DINOv3 Classifier with COMPLETE Infrastructure
## Dec 28, 2025 - ZERO MISSING FEATURES

**Status**: Complete production infrastructure from day 1
**Target**: 90%+ accuracy, ECE <0.05, 6× faster inference, full MLOps
**Total Time**: ~172 hours (~21 days / 3 weeks)
**Philosophy**: Build complete infrastructure NOW, reuse for ALL stages later

---

## EXECUTIVE SUMMARY

**WHY START WITH COMPLETE INFRASTRUCTURE:**
- Build it ONCE, use it for Stages 1, 2, 3, YOLO, FiftyOne, etc.
- Prevents technical debt
- Production-ready from day 1
- Easy to add new models/features later

**WHAT WE'RE BUILDING:**

### Core Model Features
- ✅ Multi-view inference (1 global + 3×3 tiles, batched forward pass)
- ✅ ExPLoRA (+8.2% - BIGGEST gain)
- ✅ DoRAN + Flash Attention 3 (+3%, 2× faster)
- ✅ Safe hyperparameters (dropout 0.3, WD 0.01, LS 0.1)

### Advanced Training
- ✅ 6 optimizers (AdamW, SAM2, Sophia, Muon, AdEMAMix, Schedule-Free)
- ✅ 7 loss functions (CE, Focal, LCRON, Gatekeeper, SupCon, KoLeo)
- ✅ Curriculum learning, MixUp, CutMix, AutoAugment
- ✅ FSDP2 multi-GPU training

### Complete Calibration & Metrics
- ✅ 7 calibration methods (Temperature, Beta, Dirichlet, Platt, Isotonic, Ensemble, SCRC)
- ✅ Conformal prediction (APS, RAPS, CRCP)
- ✅ All metrics (AUROC, AUPRC, ECE, AUGRC, bootstrap CI)
- ✅ Slice-based evaluation (day/night/weather)
- ✅ Drift detection (PSI, KS test, MMD)

### Production Infrastructure
- ✅ DAG pipeline (artifact registry, split contracts, zero leakage)
- ✅ MLOps (DVC, experiment tracking, monitoring)
- ✅ Deployment (ONNX, TensorRT, Docker, K8s, Triton)
- ✅ Monitoring (Prometheus, Grafana, alerts)
- ✅ Testing (unit tests, integration tests, CI/CD)
- ✅ Complete documentation

### Data Infrastructure
- ✅ Multi-dataset fusion (NATIX + Mapillary)
- ✅ Hard negative mining
- ✅ Domain adaptation (DANN)
- ✅ Active learning, pseudo-labeling

**EXPECTED RESULTS:**
- Accuracy: 69% → 90% (+21 points)
- Speed: 6× faster inference (TensorRT + batched multi-view)
- Cost: 10× cheaper training ($120 → $12 with ExPLoRA)
- Calibration: ECE 0.29 → 0.05 (-83% error)
- Production-ready: Full MLOps, monitoring, deployment

---

## COMPLETE FOLDER STRUCTURE

```
stage1_ultimate/
├── configs/                    # Hydra configs
│   ├── model/
│   │   ├── dinov2_multiview.yaml
│   │   ├── explora.yaml
│   │   ├── doran.yaml
│   │   └── flash_attention.yaml
│   ├── training/
│   │   ├── baseline.yaml
│   │   ├── optimizer_ablation.yaml      # SAM2, Sophia, Muon, etc.
│   │   ├── loss_ablation.yaml           # Focal, LCRON, SupCon, etc.
│   │   └── curriculum.yaml
│   ├── data/
│   │   ├── natix.yaml
│   │   ├── mapillary.yaml
│   │   ├── fusion.yaml
│   │   └── hard_negative.yaml
│   ├── calibration/
│   │   ├── temperature.yaml
│   │   ├── dirichlet.yaml
│   │   ├── scrc.yaml
│   │   └── conformal.yaml
│   ├── evaluation/
│   │   ├── metrics.yaml
│   │   ├── slicing.yaml
│   │   └── bootstrap.yaml
│   ├── deployment/
│   │   ├── onnx.yaml
│   │   ├── tensorrt.yaml
│   │   └── triton.yaml
│   └── config.yaml                     # Main config
│
├── src/
│   ├── contracts/              # Leakage prevention (CRITICAL)
│   │   ├── artifact_schema.py  # ⭐ Single source of truth for paths
│   │   ├── split_contracts.py  # ⭐ Split usage rules (enforced as code)
│   │   └── validators.py       # ⭐ Fail-fast checking
│   │
│   ├── pipeline/               # DAG orchestrator
│   │   ├── phase_spec.py       # Phase contracts
│   │   └── dag_engine.py       # ⭐ Dependency resolution
│   │
│   ├── models/                 # All model components
│   │   ├── module.py           # ⭐ LightningModule (main)
│   │   ├── multi_view.py       # ⭐ Multi-view inference (1+9 crops)
│   │   ├── explora.py          # ⭐ ExPLoRA (+8.2%)
│   │   ├── doran.py            # DoRAN (+1-3%)
│   │   ├── flash_attention.py  # Flash Attention 3 (1.5-2× speed)
│   │   ├── backbone.py         # DINOv3 wrapper
│   │   ├── head.py             # Classification head
│   │   ├── domain_adaptation.py # DANN discriminator
│   │   └── uncertainty.py      # 7D uncertainty features + failure gate
│   │
│   ├── data/                   # Data loading
│   │   ├── datamodule.py       # ⭐ LightningDataModule
│   │   ├── datasets.py         # NATIX, Mapillary
│   │   ├── mapillary.py        # Mapillary Vistas dataset
│   │   ├── balanced_dataset.py # Multi-dataset balancing
│   │   ├── splits.py           # 4-way split generation
│   │   ├── transforms.py       # Augmentation (MixUp, CutMix, AutoAugment)
│   │   └── hard_negative.py    # Hard negative mining
│   │
│   ├── training/               # Training logic
│   │   ├── ema.py              # EMA implementation
│   │   ├── explora_pretrain.py # ExPLoRA pretraining
│   │   ├── curriculum.py       # Curriculum learning
│   │   └── fsdp.py             # FSDP2 multi-GPU
│   │
│   ├── optimizers/             # Advanced optimizers
│   │   ├── sam2.py             # SAM2 (+1.5%)
│   │   ├── sophia.py           # Sophia (2× faster training)
│   │   ├── muon.py             # Muon (higher LR)
│   │   ├── ademamix.py         # AdEMAMix (third momentum)
│   │   └── schedule_free.py    # Schedule-Free (no LR scheduler)
│   │
│   ├── losses/                 # Advanced loss functions
│   │   ├── focal.py            # Focal Loss (class imbalance)
│   │   ├── lcron.py            # LCRON (cascade ranking)
│   │   ├── gatekeeper.py       # Gatekeeper (deferral)
│   │   ├── supcon.py           # SupCon (supervised contrastive)
│   │   └── koleo.py            # KoLeo (feature collapse prevention)
│   │
│   ├── metrics/                # Evaluation
│   │   ├── calibration.py      # ECE, MCE, ACE, Brier, NLL
│   │   ├── selective.py        # AUGRC, risk-coverage curves
│   │   ├── bootstrap.py        # Bootstrap CI (95% confidence intervals)
│   │   ├── classification.py   # AUROC, AUPRC, F1
│   │   ├── slicing.py          # Slice-based evaluation
│   │   ├── cascade.py          # Cascade metrics
│   │   └── fairness.py         # Fairness metrics
│   │
│   ├── calibration/            # Post-hoc calibration
│   │   ├── temperature.py      # Temperature scaling
│   │   ├── beta.py             # Beta calibration
│   │   ├── dirichlet.py        # Dirichlet calibration
│   │   ├── platt.py            # Platt scaling
│   │   ├── isotonic.py         # Isotonic regression
│   │   ├── ensemble.py         # Ensemble temperature
│   │   ├── scrc.py             # Split Conformal Risk Control
│   │   └── conformal.py        # APS, RAPS, CRCP
│   │
│   ├── monitoring/             # Drift detection & monitoring
│   │   ├── drift.py            # PSI, KS test, MMD
│   │   ├── prometheus.py       # Prometheus exporter
│   │   └── alerts.py           # Alert system
│   │
│   ├── deployment/             # Deployment infrastructure
│   │   ├── onnx_export.py      # ONNX export
│   │   ├── tensorrt.py         # TensorRT optimization (3-5× speedup)
│   │   ├── triton.py           # Triton Inference Server
│   │   ├── docker/
│   │   │   └── Dockerfile
│   │   ├── k8s/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── ingress.yaml
│   │   │   └── hpa.yaml
│   │   ├── ab_testing.py       # A/B testing framework
│   │   └── shadow.py           # Shadow deployment
│   │
│   ├── mlops/                  # MLOps infrastructure
│   │   ├── dvc_config.py       # DVC setup
│   │   ├── experiment_tracker.py # Experiment tracking
│   │   └── model_registry.py   # Model versioning
│   │
│   ├── evaluation/             # Evaluation reports
│   │   ├── reliability.py      # Reliability diagrams
│   │   ├── calibration_summary.py # Calibration comparison
│   │   ├── summary_report.py   # Model comparison
│   │   └── cross_dataset.py    # Cross-dataset validation
│   │
│   └── utils/
│       ├── logging.py
│       └── visualization.py
│
├── scripts/                    # CLI entry points
│   ├── 10_train_baseline.py
│   ├── 15_train_explora.py
│   ├── 20_calibrate.py
│   ├── 30_evaluate.py
│   ├── 40_export.py
│   └── train_cli.py            # Main CLI (DAG engine)
│
├── tests/                      # Testing infrastructure
│   ├── unit/
│   │   ├── test_multi_view.py
│   │   ├── test_calibration.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_dag_pipeline.py
│   │   ├── test_training.py
│   │   └── test_deployment.py
│   └── conftest.py
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── CONTRIBUTING.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI/CD pipeline
│       └── deploy.yml
│
├── deployment/
│   ├── monitoring/
│   │   └── grafana/
│   │       └── dashboards/
│   └── docker-compose.yml
│
├── train.py                    # Main entry point
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .dvcignore
├── .gitignore
└── README.md
```

---

## IMPLEMENTATION TIMELINE (21 Days / 172 Hours)

### Week 1: Core Infrastructure (Days 1-7, 56h)

**Tier 0: DAG Pipeline Architecture (Days 1-2, 14h) - TODOs 121-140**
- ✅ Artifact registry (single source of truth for paths)
- ✅ Split contracts (zero data leakage, enforced as code)
- ✅ Validators (fail-fast artifact checking)
- ✅ Phase specifications (DAG nodes with contracts)
- ✅ DAG engine (automatic dependency resolution)
- ✅ Clean CLI entry point
- ✅ Hydra configuration structure
- ✅ Integration tests

**Foundation (Days 3-4, 12h) - TODOs 1-20**
- ✅ File cleanup (remove duplicates)
- ✅ Fix trainer call signatures
- ✅ Implement SCRC stub methods
- ✅ Multi-view generator (1 global + 9 tiles, 15% overlap)
- ✅ Top-K mean aggregator (K=2 or 3)
- ✅ Attention aggregator

**Core Model (Days 5-7, 30h) - TODOs 141-160 (SOTA Features)**
- ✅ ExPLoRA implementation (+8.2% - BIGGEST gain)
- ✅ DoRAN head (+1-3% over LoRA)
- ✅ Flash Attention 3 (1.5-2× faster)
- ✅ Lightning Module + DataModule
- ✅ EMA implementation
- ✅ 7D uncertainty features
- ✅ Failure gate predictor (AUROC 0.85)

---

### Week 2: Training & Calibration (Days 8-14, 68h)

**Advanced Training (Days 8-9, 16h) - TODOs 31-50**
- ✅ 6 optimizer implementations (SAM2, Sophia, Muon, AdEMAMix, Schedule-Free, AdamW)
- ✅ 7 loss function implementations (CE, Focal, LCRON, Gatekeeper, SupCon, KoLeo)
- ✅ Curriculum learning
- ✅ MixUp, CutMix, AutoAugment
- ✅ FSDP2 multi-GPU training

**Complete Calibration (Days 10-11, 24h) - TODOs 161-180**
- ✅ Temperature scaling (LBFGS optimization, -50% ECE)
- ✅ Beta calibration (MLE fitting, -65% ECE)
- ✅ Class-wise temperature (-60% ECE)
- ✅ Platt scaling (SGD optimization, -50% ECE)
- ✅ Isotonic regression (-55% ECE)
- ✅ Ensemble temperature (-70% ECE)
- ✅ Dirichlet calibration (-60% ECE)
- ✅ Calibration by slice
- ✅ Reliability diagram generator
- ✅ Calibration summary report

**Complete Evaluation (Days 12-13, 14h) - TODOs 51-70 + 171-180**
- ✅ AUROC/AUPRC computation
- ✅ Precision/Recall/F1
- ✅ ECE/MCE/SCE computation
- ✅ Brier score, NLL
- ✅ Risk-coverage curve (AUGRC)
- ✅ Coverage-at-risk, risk-at-coverage
- ✅ Cascade metrics
- ✅ Fairness metrics
- ✅ Slice-based evaluation (day/night/weather)
- ✅ Evaluation summary report

**Drift Detection & Tuning (Day 14, 14h) - TODOs 86-110**
- ✅ Bootstrap confidence intervals (95% CI)
- ✅ PSI (Population Stability Index)
- ✅ KS test (Kolmogorov-Smirnov)
- ✅ MMD (Maximum Mean Discrepancy)
- ✅ Embedding shift detection
- ✅ Hyperparameter sweep grids (LR, batch size, LoRA rank)
- ✅ Bayesian optimization (Optuna/TPE)

---

### Week 3: Production Infrastructure (Days 15-21, 48h)

**Deployment Infrastructure (Days 15-16, 15h) - TODOs 181-195**
- ✅ ONNX export (3.5× speedup)
- ✅ TensorRT optimization (3-5× speedup)
- ✅ Triton Inference Server (production serving)
- ✅ Docker containerization
- ✅ Kubernetes deployment manifests (deployment, service, ingress, HPA)
- ✅ Prometheus metrics exporter
- ✅ Grafana dashboards
- ✅ A/B testing framework
- ✅ Shadow deployment
- ✅ Monitoring & alerting
- ✅ Load testing, model registry, versioning, rollback

**Multi-Dataset Fusion (Days 17-18, 10h) - TODOs 111-120 + 196-210**
- ✅ Mapillary Vistas integration (25K images, 21GB)
- ✅ Dataset balancing (50/50 or 30/70 NATIX/Mapillary)
- ✅ Domain adaptation (DANN discriminator)
- ✅ Hard negative mining (orange objects, construction signs)
- ✅ Cross-dataset validation
- ✅ Pseudo-labeling, active learning
- ✅ Multi-dataset calibration
- ✅ CutMix across datasets
- ✅ Dataset performance analysis

**Testing & Documentation (Days 19-21, 23h)**
- ✅ Unit tests (multi-view, calibration, metrics)
- ✅ Integration tests (DAG pipeline, training, deployment)
- ✅ CI/CD pipeline (.github/workflows/)
- ✅ Architecture documentation
- ✅ API reference
- ✅ Deployment guide
- ✅ End-to-end pipeline test
- ✅ Accuracy verification (88-92%+)
- ✅ Speed benchmarking (>30 FPS)
- ✅ Production readiness checklist
- ✅ Complete documentation review

---

## COMPLETE 210-TODO CHECKLIST

### Tier 0: DAG Pipeline Architecture (14h) - TODOs 121-140 ⭐⭐⭐

- [ ] **TODO 121**: Create `contracts/artifact_schema.py` - Artifact Registry (1.5h)
- [ ] **TODO 122**: Create `contracts/split_contracts.py` - Leakage Prevention (1h)
- [ ] **TODO 123**: Create `contracts/validators.py` - Fail-Fast Checking (2h)
- [ ] **TODO 124**: Create `pipeline/phase_spec.py` - DAG Phase Specifications (2.5h)
- [ ] **TODO 125**: Create `pipeline/dag_engine.py` - DAG Pipeline Orchestrator (2h)
- [ ] **TODO 126**: Create `scripts/train_cli.py` - Clean CLI Entry Point (1h)
- [ ] **TODO 127**: Create base config structure with Hydra (1h)
- [ ] **TODO 128**: Create phase-specific configs (1h)
- [ ] **TODO 129**: Update existing code to use ArtifactSchema (1h)
- [ ] **TODO 130**: Add integration test for DAG pipeline (1h)

### Foundation (12h) - TODOs 1-20

- [ ] **TODO 1-5**: Cleanup & Fixes (2h)
  - Delete duplicate files (peft.py, peft_custom.py, calibrate_gate.py)
  - Fix scripts/20_train.py trainer call
  - Implement calibration/scrc.py methods
- [ ] **TODO 6-10**: Multi-View Generator Schema (3h)
- [ ] **TODO 11-15**: Multi-View Aggregation (3h)
- [ ] **TODO 16-20**: Multi-View Integration & Testing (4h)

### Tier 1: SOTA Features (28h) - TODOs 141-160 ⭐⭐⭐

- [ ] **TODO 141**: Create `models/explora.py` - ExPLoRA PEFT (2.5h) **+8.2%**
- [ ] **TODO 142**: Create `models/doran_head.py` - DoRAN PEFT (2.5h) **+1-3%**
- [ ] **TODO 143**: Create `models/flash_attn3.py` - Flash Attention 3 (2h) **1.5-2× speed**
- [ ] **TODO 144**: Create `models/multi_view.py` - Multi-View Inference (3h) **+3-5%**
- [ ] **TODO 145**: Create `models/uncertainty.py` - 7D Uncertainty Features (3h)
- [ ] **TODO 146**: Create `models/failure_gate.py` - Failure Predictor (2.5h) **AUROC 0.85**
- [ ] **TODO 147**: Create `losses/lcron.py` - LCRON Loss (2h) **+3.5%**
- [ ] **TODO 148**: Create `losses/gatekeeper.py` - Gatekeeper Loss (1.5h) **+2.3%**
- [ ] **TODO 149**: Hard negative mining (1.5h) **+2%**
- [ ] **TODO 150**: Hierarchical validation (1h)
- [ ] **TODO 151**: torch.compile integration (1h) **1.3-2× speed**
- [ ] **TODO 152**: Mixed precision training (1h) **2× memory**
- [ ] **TODO 153**: Gradient checkpointing (1h) **3× memory**
- [ ] **TODO 154**: FSDP2 multi-GPU (1.5h) **2× memory reduction**
- [ ] **TODO 155**: Curriculum learning (1.5h) **+1-2%**
- [ ] **TODO 156**: Advanced augmentation (MixUp, CutMix, AutoAugment) (1.5h) **+1-2%**
- [ ] **TODO 157**: Hierarchical Stochastic Attention (1h)
- [ ] **TODO 158**: Domain discriminator (DANN) (1.5h) **+1-2%**
- [ ] **TODO 159**: SCRC/CRCP implementation (1.5h)
- [ ] **TODO 160**: Integration testing for SOTA features (1h)

### Advanced Training (16h) - TODOs 31-50

- [ ] **TODO 31-40**: Optimizer Ablation (8h)
  - AdamW baseline
  - SAM2 (+1.5%, 2× slower)
  - Sophia (+1%, 2× FASTER)
  - Schedule-Free (no LR scheduler)
  - AdEMAMix (+0.5%)
  - Muon (+1.5%)
- [ ] **TODO 41-50**: Loss Function Ablation (8h)
  - Cross-entropy baseline
  - Focal Loss (+1% if imbalanced)
  - LCRON (+3.5% cascade recall)
  - Gatekeeper (+2.3% deferral)
  - SupCon (+1.5%)
  - KoLeo (+0.5% stability)
  - Combined losses

### Evaluation (12h) - TODOs 51-70

- [ ] **TODO 51-60**: Evaluation Metrics Schema (6h)
  - AUROC, AUPRC, F1
  - ECE, MCE, SCE, Brier, NLL
  - AUGRC, risk-coverage curves
- [ ] **TODO 61-70**: Slice-Based Evaluation (6h)
  - Day/night/dawn/dusk slices
  - Weather slices (clear/rain/snow/fog)
  - Camera source slices
  - Confidence bin slices

### Calibration (10h) - TODOs 71-85

- [ ] **TODO 71-80**: Calibration Methods (6h)
  - Temperature scaling (-50% ECE)
  - Class-wise temperature (-60% ECE)
  - Platt scaling (-50% ECE)
  - Beta calibration (-65% ECE)
  - Isotonic regression (-55% ECE)
  - Ensemble temperature (-70% ECE)
  - Dirichlet calibration (-60% ECE)
- [ ] **TODO 81-85**: Conformal Prediction (4h)
  - Split conformal
  - SCRC (robust to contamination)
  - CRCP (zero-shot models)
  - APS (adaptive prediction sets)
  - RAPS (regularized APS)

### Tier 2: Calibration & Evaluation Implementation (24h) - TODOs 161-180

- [ ] **TODO 161-170**: Calibration Implementation (10h)
  - Full code for all 7 calibration methods
  - Reliability diagram generator
  - Calibration summary report
- [ ] **TODO 171-180**: Evaluation Implementation (14h)
  - AUROC/AUPRC computation
  - Precision/Recall/F1 computation
  - ECE/MCE/SCE computation
  - Brier score, NLL
  - Risk-coverage curves
  - Cascade metrics
  - Fairness metrics
  - Evaluation summary report

### Bootstrap & Drift (8h) - TODOs 86-95

- [ ] **TODO 86-90**: Bootstrap Confidence Intervals (4h)
  - 1000 bootstrap resamples
  - 95% confidence intervals
  - Statistical significance testing
- [ ] **TODO 91-95**: Drift Detection (4h)
  - PSI (Population Stability Index)
  - KS test (Kolmogorov-Smirnov)
  - MMD (Maximum Mean Discrepancy)
  - Embedding shift detection

### Hyperparameter Tuning (10h) - TODOs 96-110

- [ ] **TODO 96-105**: Hyperparameter Sweep Grids (6h)
  - LR sweep: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
  - Weight decay sweep: [0.0, 0.01, 0.05, 0.1, 0.5]
  - Batch size sweep: [8, 16, 32, 64, 128]
  - LoRA rank sweep: [4, 8, 16, 32, 64]
  - Dropout sweep: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- [ ] **TODO 106-110**: Bayesian Optimization (4h)
  - Optuna/TPE sampler
  - Multi-parameter search
  - Objective: val_select/accuracy

### Data Fusion (8h) - TODOs 111-120

- [ ] **TODO 111-120**: Multi-Dataset Fusion Schema (8h)
  - Naive concatenation
  - Balanced sampling (50/50)
  - Weighted loss (inverse frequency)
  - Domain stratified (equal per domain)
  - Hard negative focused (+2.3%)
  - Dataset mixing ratios (30/70 recommended)
  - Class balancing (WeightedRandomSampler)
  - Data quality checks (duplicates, outliers, label consistency)

### Tier 3: Deployment (15h) - TODOs 181-195 ⭐⭐

- [ ] **TODO 181**: ONNX Export (1h) **3.5× speedup**
- [ ] **TODO 182**: TensorRT Optimization (2h) **3-5× speedup**
- [ ] **TODO 183**: Triton Inference Server (2h)
- [ ] **TODO 184**: Docker Containerization (1.5h)
- [ ] **TODO 185**: Kubernetes Deployment Manifests (1.5h)
- [ ] **TODO 186**: Prometheus Metrics Exporter (1.5h)
- [ ] **TODO 187**: Grafana Dashboards (1.5h)
- [ ] **TODO 188**: A/B Testing Framework (2h)
- [ ] **TODO 189**: Shadow Deployment (1.5h)
- [ ] **TODO 190**: Monitoring & Alerting (1h)
- [ ] **TODO 191-195**: Additional Deployment (5h)
  - Load testing (Locust)
  - Model registry (MLflow)
  - Versioning system
  - Rollback mechanism
  - Production checklist

### Tier 4: Multi-Dataset Fusion (10h) - TODOs 196-210

- [ ] **TODO 196**: Mapillary Vistas Integration (2h) **25K images, 21GB**
- [ ] **TODO 197**: Dataset Balancing (50/50 or 30/70) (1.5h) **+2-3%**
- [ ] **TODO 198**: Domain Adaptation (DANN) (2h) **+1-2%**
- [ ] **TODO 199**: Hard Negative Mining from Mapillary (1.5h)
- [ ] **TODO 200**: Cross-dataset Validation (1h)
- [ ] **TODO 201-205**: Additional Fusion Components (5h)
  - Pseudo-labeling unlabeled Mapillary
  - Active learning
  - Multi-dataset calibration
  - CutMix across datasets
  - Dataset performance analysis
- [ ] **TODO 206-210**: Final Validation (5h)
  - End-to-end pipeline test
  - Accuracy verification (88-92%+)
  - Speed benchmarking (>30 FPS)
  - Production readiness checklist
  - Complete documentation review

---

## SUCCESS CRITERIA

**Must achieve ALL of these:**

1. ✅ **Accuracy**: ≥88% on validation set (target: 90%)
2. ✅ **Calibration**: ECE ≤0.10 (target: 0.05)
3. ✅ **Speed**: ≥30 FPS inference on single GPU (≥60 FPS with TensorRT)
4. ✅ **Cost**: Training cost ≤$15 (with ExPLoRA)
5. ✅ **Leakage**: Zero data leakage (enforced by split contracts)
6. ✅ **Multi-view**: Batched forward pass working correctly
7. ✅ **Architecture**: Complete DAG pipeline with artifact registry
8. ✅ **Metrics**: All metrics implemented (AUROC, ECE, bootstrap CI, slicing)
9. ✅ **Calibration**: All 7 methods + conformal prediction working
10. ✅ **Deployment**: ONNX, TensorRT, Docker, K8s ready
11. ✅ **Monitoring**: Prometheus, Grafana, alerts working
12. ✅ **Testing**: All unit + integration tests passing
13. ✅ **Documentation**: Complete architecture + API docs
14. ✅ **MLOps**: DVC, experiment tracking, model registry working

---

## NEXT STEPS

**Day 1: Start with Tier 0 (DAG Pipeline Architecture)**

1. ✅ Create `src/contracts/artifact_schema.py`
2. ✅ Create `src/contracts/split_contracts.py`
3. ✅ Create `src/contracts/validators.py`
4. ✅ Create `src/pipeline/phase_spec.py`
5. ✅ Create `src/pipeline/dag_engine.py`
6. ✅ Create `scripts/train_cli.py`
7. ✅ Create Hydra configs
8. ✅ Test DAG pipeline

This gives us ZERO data leakage and solid foundation for everything else!

**Ready to start implementation!** 🚀
