# 🚀 **TODO IMPLEMENTATION ORDER - 2025/2026 PROJECT**
## Following olanzapin.md 210-TODO plan

***

## 📋 **IMPLEMENTATION PHASES (DO IN ORDER)**

### **TIER 0: FOUNDATION & CLI (Start Here - 6h)** ⭐⭐⭐
**GOAL**: Working CLI that can train a simple model

#### **Phase 0.1: Project Setup (1h)**
```yaml
status: READY_TO_START
prerequisites: None

tasks:
  - TODO_001: Create project structure (src/, configs/, tests/)
  - TODO_002: Setup pyproject.toml with Poetry
  - TODO_003: Create Pydantic 2.9 configs (src/configs/config.py)
  - TODO_004: Create .env file for secrets
  - TODO_005: Setup pre-commit hooks

expected_output:
  - Working Poetry project
  - Validated configs
  - Pre-commit hooks installed
```

#### **Phase 0.2: Data Module (1.5h)**
```yaml
status: BLOCKED_BY Phase 0.1
prerequisites: TODO_001, TODO_003

tasks:
  - TODO_006: Create BaseDataModule (LightningDataModule)
  - TODO_007: Create NATIXDataset with timm transforms
  - TODO_008: Create MultiRoadworkDataset (NATIX + ROADWork + Mapillary)
  - TODO_009: Implement weighted random sampler (class balancing)
  - TODO_010: Create split validation (train/val_select/val_calib/val_test)

expected_output:
  - Working datamodule
  - Can load and iterate over batches
  - Proper split validation (no leakage)
```

#### **Phase 0.3: Model Module (1.5h)**
```yaml
status: BLOCKED_BY Phase 0.2
prerequisites: TODO_006

tasks:
  - TODO_011: Create BaseModel (LightningModule)
  - TODO_012: Create DINOv3Backbone wrapper
  - TODO_013: Create Stage1Head (simple classification)
  - TODO_014: Integrate torch.compile (30-50% speedup)
  - TODO_015: Create MultiViewInference module

expected_output:
  - Working model with DINOv3 backbone
  - Compiled model (faster)
  - Multi-view support
```

#### **Phase 0.4: CLI Entry Point (1h)**
```yaml
status: BLOCKED_BY Phase 0.3
prerequisites: TODO_011, TODO_015

tasks:
  - TODO_016: Create scripts/train.py (Lightning CLI)
  - TODO_017: Create configs/hydra/ (phase1.yaml, phase3.yaml, etc.)
  - TODO_018: Setup W&B logging
  - TODO_019: Setup MLflow tracking
  - TODO_020: Create smoke test (1 epoch on small data)

expected_output:
  - Can run: python scripts/train.py --config configs/hydra/phase1.yaml
  - Logs appear in W&B/MLflow
  - Trains for 1 epoch successfully
```

---

### **TIER 1: SOTA FEATURES - ExPLoRA, DoRAN, Flash Attention (8h)**
**GOAL**: +8.2% accuracy with ExPLoRA

#### **Phase 1.1: ExPLoRA Implementation (2.5h)** ⭐ +8.2%
```yaml
status: BLOCKED_BY Phase 0.4
prerequisites: TODO_020

tasks:
  - TODO_021: Create ExPLoRAConfig (extends PEFT config)
  - TODO_022: Implement ExPLoRA pretraining loop (Stage 1)
  - TODO_023: Implement supervised fine-tuning loop (Stage 2)
  - TODO_024: Create scripts/train_explora.py
  - TODO_025: Test ExPLoRA on small dataset

expected_output:
  - ExPLoRA pretraining works
  - Fine-tuning improves accuracy by +5-8%
  - Documentation: expected +8.2% gain
```

#### **Phase 1.2: DoRAN Implementation (2h)** ⭐ +1-3%
```yaml
status: BLOCKED_BY Phase 1.1
prerequisites: TODO_025

tasks:
  - TODO_026: Create DoRANLinear layer (decomposed LoRA + noise)
  - TODO_027: Create DoRANConfig
  - TODO_028: Integrate DoRAN with PEFT library
  - TODO_029: Compare DoRAN vs LoRA vs standard
  - TODO_030: Document expected +1-3% gain

expected_output:
  - DoRAN more stable than DoRA
  - +1-3% accuracy over LoRA
  - Comparison metrics saved
```

#### **Phase 1.3: Flash Attention 3 (1.5h)** ⭐ 1.5-2× SPEED
```yaml
status: BLOCKED_BY Phase 1.2
prerequisites: TODO_030

tasks:
  - TODO_031: Install flash-attn==3.0.0
  - TODO_032: Modify DINOv3Backbone to use Flash Attention
  - TODO_033: Benchmark: standard vs flash attention
  - TODO_034: Update configs (use_flash_attn flag)
  - TODO_035: Test on H100/A100

expected_output:
  - 1.5-2× speedup on H100
  - Benchmark results
  - Config flag works
```

#### **Phase 1.4: torch.compile Integration (1h)** ⭐ 30-50% FREE SPEEDUP
```yaml
status: BLOCKED_BY Phase 1.3
prerequisites: TODO_032

tasks:
  - TODO_036: Enable torch.compile for all models
  - TODO_037: Test compile modes: default, reduce-overhead, max-autotune
  - TODO_038: Benchmark compiled vs uncompiled
  - TODO_039: Add compile options to configs
  - TODO_040: Document expected 30-50% speedup

expected_output:
  - All models compiled
  - 30-50% speedup (FREE)
  - Benchmark results
```

---

### **TIER 2: MULTI-VIEW INFERENCE (6h)**
**GOAL**: +3-5% accuracy with multi-view

#### **Phase 2.1: Multi-View Generator (2h)**
```yaml
status: BLOCKED_BY Phase 1.4
prerequisites: TODO_040

tasks:
  - TODO_041: Create MultiViewGenerator (1 global + 9 tiles)
  - TODO_042: Implement batched forward pass (7× speedup)
  - TODO_043: Create AttentionAggregator
  - TODO_044: Create TopKMeanAggregator (K=2,3,5)
  - TODO_045: Add TTA support (horizontal flip)

expected_output:
  - Multi-view generation works
  - Batched inference 7× faster
  - 3 aggregation methods implemented
```

#### **Phase 2.2: Multi-View Training (2h)**
```yaml
status: BLOCKED_BY Phase 2.1
prerequisites: TODO_045

tasks:
  - TODO_046: Modify BaseModel to support multi-view
  - TODO_047: Create multi-view training loop
  - TODO_048: Add multi-view data augmentation
  - TODO_049: Test multi-view vs single-view
  - TODO_050: Document expected +3-5% gain

expected_output:
  - Multi-view training works
  - +3-5% accuracy gain
  - Comparison metrics
```

#### **Phase 2.3: Multi-View Inference (2h)**
```yaml
status: BLOCKED_BY Phase 2.2
prerequisites: TODO_050

tasks:
  - TODO_051: Create scripts/infer_multiview.py
  - TODO_052: Optimize inference (batched crops)
  - TODO_053: Add visualization (attention weights)
  - TODO_054: Benchmark inference speed
  - TODO_055: Test on real data

expected_output:
  - Multi-view inference works
  - Visualization of attention
  - Inference speed benchmarks
```

---

### **TIER 3: ADVANCED TRAINING - LCRON, Gatekeeper, SAM (8h)**
**GOAL**: Better cascade calibration and training

#### **Phase 3.1: LCRON Loss (2.5h)** ⭐ +3-5% CASCADE
```yaml
status: BLOCKED_BY Phase 2.3
prerequisites: TODO_055

tasks:
  - TODO_056: Implement LCRONLoss (3 objectives)
  - TODO_057: Create LCRONCascade model
  - TODO_058: Train with LCRON vs standard CE
  - TODO_059: Compare cascade metrics
  - TODO_060: Document expected +3-5% cascade recall

expected_output:
  - LCRON loss implemented
  - Better cascade recall @90% precision
  - Comparison metrics
```

#### **Phase 3.2: Gatekeeper Calibration (2h)** ⭐ +2-3% CALIBRATION
```yaml
status: BLOCKED_BY Phase 3.1
prerequisites: TODO_060

tasks:
  - TODO_061: Implement GatekeeperCalibration
  - TODO_062: Create calibration map (per confidence bin)
  - TODO_063: Fit calibration on val_calib
  - TODO_064: Test calibration on val_test
  - TODO_065: Document expected +2-3% ECE improvement

expected_output:
  - Gatekeeper calibration works
  - +2-3% ECE improvement
  - Calibration visualizations
```

#### **Phase 3.3: SAM Optimizer (2h)** ⭐ +1-2%
```yaml
status: BLOCKED_BY Phase 3.2
prerequisites: TODO_065

tasks:
  - TODO_066: Install SAM (pip install sam-py)
  - TODO_067: Integrate SAM with Lightning
  - TODO_068: Train with SAM vs AdamW
  - TODO_069: Compare accuracy and training time
  - TODO_070: Document expected +1-2% accuracy

expected_output:
  - SAM optimizer works
  - Better generalization
  - Comparison metrics
```

#### **Phase 3.4: Additional Optimizers (1.5h)**
```yaml
status: BLOCKED_BY Phase 3.3
prerequisites: TODO_070

tasks:
  - TODO_071: Implement Sophia optimizer
  - TODO_072: Implement ScheduleFree optimizer
  - TODO_073: Implement AdEMAMix optimizer
  - TODO_074: Benchmark all optimizers
  - TODO_075: Select best optimizer for production

expected_output:
  - All optimizers work
  - Benchmark results
  - Best optimizer selected
```

---

### **TIER 4: CALIBRATION & EVALUATION (10h)**
**GOAL**: Robust calibration and comprehensive evaluation

#### **Phase 4.1: Calibration Methods (5h)**
```yaml
status: BLOCKED_BY Phase 3.4
prerequisites: TODO_075

tasks:
  - TODO_076: Implement TemperatureScaling (LBFGS)
  - TODO_077: Implement BetaCalibration (MLE)
  - TODO_078: Implement IsotonicRegression
  - TODO_079: Implement DirichletCalibration
  - TODO_080: Compare all calibration methods
  - TODO_081: Select best method (expected 50-70% ECE reduction)

expected_output:
  - All calibration methods work
  - ECE reduced by 50-70%
  - Reliability diagrams
```

#### **Phase 4.2: Conformal Prediction (2h)**
```yaml
status: BLOCKED_BY Phase 4.1
prerequisites: TODO_081

tasks:
  - TODO_082: Implement SplitConformal
  - TODO_083: Implement SCRC (robust to label noise)
  - TODO_084: Implement CRCP (ranking-based)
  - TODO_085: Implement APS and RAPS
  - TODO_086: Verify 90% coverage

expected_output:
  - All conformal methods work
  - 90% coverage guaranteed
  - Prediction sets generated
```

#### **Phase 4.3: Evaluation Metrics (3h)**
```yaml
status: BLOCKED_BY Phase 4.2
prerequisites: TODO_086

tasks:
  - TODO_087: Compute AUROC/AUPRC
  - TODO_088: Compute ECE/MCE/SCE
  - TODO_089: Compute Brier score and NLL
  - TODO_090: Compute risk-coverage curve (AUGRC)
  - TODO_091: Compute slice-based metrics (day/night/weather)
  - TODO_092: Create evaluation report

expected_output:
  - All metrics computed
  - Slice-based evaluation
  - Comprehensive report
```

---

### **TIER 5: DAG PIPELINE & CONTRACTS (10h)**
**GOAL**: Zero leakage, fail-fast validation, automatic dependency resolution

#### **Phase 5.1: Artifact Registry (2h)**
```yaml
status: BLOCKED_BY Phase 4.3
prerequisites: TODO_092

tasks:
  - TODO_093: Create ArtifactSchema (Pydantic)
  - TODO_094: Define all artifact paths
  - TODO_095: Implement get_required_inputs()
  - TODO_096: Implement get_expected_outputs()
  - TODO_097: Add artifact validation

expected_output:
  - Single source of truth for paths
  - Artifacts validated on save/load
  - Clear contract for each phase
```

#### **Phase 5.2: Split Contracts (2h)**
```yaml
status: BLOCKED_BY Phase 5.1
prerequisites: TODO_097

tasks:
  - TODO_098: Create Split enum (TRAIN, VAL_SELECT, VAL_CALIB, VAL_TEST)
  - TODO_099: Implement validate_model_selection()
  - TODO_100: Implement validate_policy_fitting()
  - TODO_101: Implement validate_final_eval()
  - TODO_102: Add enforcement checks

expected_output:
  - Split usage enforced
  - Zero data leakage by construction
  - Clear validation rules
```

#### **Phase 5.3: DAG Engine (3h)**
```yaml
status: BLOCKED_BY Phase 5.2
prerequisites: TODO_102

tasks:
  - TODO_103: Create PhaseSpec (ABC)
  - TODO_104: Implement Phase1Spec through Phase6Spec
  - TODO_105: Create DAGEngine
  - TODO_106: Implement resolve_dependencies()
  - TODO_107: Implement run_phase()
  - TODO_108: Test dependency resolution

expected_output:
  - DAG engine works
  - Automatic dependency resolution
  - Can run phases independently or together
```

#### **Phase 5.4: Integration (3h)**
```yaml
status: BLOCKED_BY Phase 5.3
prerequisites: TODO_108

tasks:
  - TODO_109: Update existing code to use ArtifactSchema
  - TODO_110: Update BaseModel to validate splits
  - TODO_111: Update CLI to use DAG engine
  - TODO_112: Create end-to-end test
  - TODO_113: Verify zero leakage

expected_output:
  - All phases use DAG engine
  - No split violations
  - End-to-end pipeline works
```

---

### **TIER 6: MLOPS INFRASTRUCTURE (8h)**
**GOAL**: Reproducible experiments, model registry, CI/CD

#### **Phase 6.1: Experiment Tracking (2h)**
```yaml
status: BLOCKED_BY Phase 5.4
prerequisites: TODO_113

tasks:
  - TODO_114: Setup W&B logging
  - TODO_115: Setup MLflow tracking
  - TODO_116: Log all metrics, artifacts, configs
  - TODO_117: Create experiment comparison UI
  - TODO_118: Document reproducibility

expected_output:
  - All experiments tracked
  - Easy to compare runs
  - Reproducible configs
```

#### **Phase 6.2: DVC Pipeline (2h)**
```yaml
status: BLOCKED_BY Phase 6.1
prerequisites: TODO_118

tasks:
  - TODO_119: Initialize DVC repo
  - TODO_120: Track data with DVC
  - TODO_121: Create DVC pipeline (dvc.yaml)
  - TODO_122: Setup cloud storage (S3/GCS)
  - TODO_123: Test DVC pipeline

expected_output:
  - Data versioned with DVC
  - Reproducible pipeline
  - Can reproduce any run
```

#### **Phase 6.3: CI/CD (2h)**
```yaml
status: BLOCKED_BY Phase 6.2
prerequisites: TODO_123

tasks:
  - TODO_124: Create GitHub Actions workflow
  - TODO_125: Add pre-commit hooks (ruff, black, pytest)
  - TODO_126: Add type checking (pyright)
  - TODO_127: Add unit tests with coverage
  - TODO_128: Auto-deploy on main branch

expected_output:
  - CI pipeline works
  - Code quality enforced
  - Tests run on every PR
```

#### **Phase 6.4: Monitoring (2h)**
```yaml
status: BLOCKED_BY Phase 6.3
prerequisites: TODO_128

tasks:
  - TODO_129: Setup Prometheus metrics
  - TODO_130: Create Grafana dashboards
  - TODO_131: Add alerting rules
  - TODO_132: Setup drift detection (PSI/KS/MMD)
  - TODO_133: Test monitoring pipeline

expected_output:
  - Real-time metrics
  - Dashboards visualized
  - Alerts configured
```

---

### **TIER 7: DEPLOYMENT (8h)**
**GOAL**: Production-ready model serving

#### **Phase 7.1: ONNX Export (2h)**
```yaml
status: BLOCKED_BY Phase 6.4
prerequisites: TODO_133

tasks:
  - TODO_134: Create export_to_onnx() function
  - TODO_135: Export all model variants
  - TODO_136: Validate ONNX models
  - TODO_137: Benchmark ONNX vs PyTorch
  - TODO_138: Document 3.5× speedup

expected_output:
  - Models exported to ONNX
  - 3.5× inference speedup
  - Benchmark results
```

#### **Phase 7.2: TensorRT Optimization (2h)** ⭐ 3-5× SPEED
```yaml
status: BLOCKED_BY Phase 7.1
prerequisites: TODO_138

tasks:
  - TODO_139: Install TensorRT
  - TODO_140: Build TensorRT engines
  - TODO_141: Benchmark TensorRT vs ONNX
  - TODO_142: Optimize for H100
  - TODO_143: Document 3-5× speedup

expected_output:
  - TensorRT engines built
  - 3-5× speedup vs ONNX
  - H100 optimized
```

#### **Phase 7.3: Docker & Kubernetes (2h)**
```yaml
status: BLOCKED_BY Phase 7.2
prerequisites: TODO_143

tasks:
  - TODO_144: Create Dockerfile (multi-stage)
  - TODO_145: Create K8s manifests
  - TODO_146: Add health checks
  - TODO_147: Setup autoscaling (KEDA)
  - TODO_148: Test deployment

expected_output:
  - Docker image works
  - K8s deployment successful
  - Autoscaling configured
```

#### **Phase 7.4: Triton Inference Server (2h)**
```yaml
status: BLOCKED_BY Phase 7.3
prerequisites: TODO_148

tasks:
  - TODO_149: Create Triton config.pbtxt
  - TODO_150: Deploy Triton server
  - TODO_151: Load TensorRT engine
  - TODO_152: Benchmark throughput
  - TODO_153: Document QPS

expected_output:
  - Triton server running
  - 1000+ QPS on A100
  - Production-ready
```

---

### **TIER 8: MULTI-DATASET FUSION (6h)**
**GOAL**: +2-3% accuracy with Mapillary integration

#### **Phase 8.1: Mapillary Integration (2h)**
```yaml
status: BLOCKED_BY Phase 7.4
prerequisites: TODO_153

tasks:
  - TODO_154: Download Mapillary Vistas dataset
  - TODO_155: Extract roadwork classes (19, 20, 55)
  - TODO_156: Create MapillaryVistasDataset
  - TODO_157: Preprocess Mapillary data
  - TODO_158: Add to multi-dataset fusion

expected_output:
  - Mapillary dataset loaded
  - 25K roadwork images added
  - Works with MultiRoadworkDataset
```

#### **Phase 8.2: Domain Adaptation (2h)**
```yaml
status: BLOCKED_BY Phase 8.1
prerequisites: TODO_158

tasks:
  - TODO_159: Create DANN (Domain Adversarial Neural Network)
  - TODO_160: Create DomainDiscriminator
  - TODO_161: Implement GRL (Gradient Reversal Layer)
  - TODO_162: Train with DANN vs standard
  - TODO_163: Document +1-2% generalization

expected_output:
  - DANN works
  - Better generalization
  - +1-2% accuracy
```

#### **Phase 8.3: Fusion & Balancing (2h)**
```yaml
status: BLOCKED_BY Phase 8.2
prerequisites: TODO_163

tasks:
  - TODO_164: Implement 50/50 balancing (NATIX/Mapillary)
  - TODO_165: Implement hard negative mining
  - TODO_166: Train with fusion strategies
  - TODO_167: Compare fusion vs single-dataset
  - TODO_168: Document +2-3% gain

expected_output:
  - Balanced multi-dataset
  - +2-3% accuracy
  - Comparison metrics
```

---

## 📋 **FINAL CHECKLIST**

### **Before Starting (Pre-requisites)**
- [ ] GPU available (H100/A100/RTX 4090)
- [ ] CUDA 12.6 installed
- [ ] Python 3.12 installed
- [ ] Poetry installed
- [ ] Git initialized

### **After Each Phase**
- [ ] Tests pass (pytest)
- [ ] Type checking passes (pyright)
- [ ] Linting passes (ruff)
- [ ] Documentation updated
- [ ] W&B/MLflow logged

### **Before Production**
- [ ] All 168 TODOs complete
- [ ] End-to-end test passes
- [ ] Accuracy ≥ 88% target
- [ ] Inference speed ≥ 30 FPS
- [ ] Calibration ECE ≤ 0.05
- [ ] Deployment tested on staging

---

## 🎯 **EXPECTED RESULTS**

| TIER | Time | Expected Gains |
|------|------|---------------|
| **TIER 0: Foundation** | 6h | Working CLI, basic training |
| **TIER 1: SOTA** | 8h | +8.2% accuracy (ExPLoRA), 2× speed (FA3 + compile) |
| **TIER 2: Multi-View** | 6h | +3-5% accuracy |
| **TIER 3: Advanced Training** | 8h | +3-5% cascade recall, +1-2% accuracy |
| **TIER 4: Calibration** | 10h | 50-70% ECE reduction, 90% coverage |
| **TIER 5: DAG** | 10h | Zero leakage, validated artifacts |
| **TIER 6: MLOps** | 8h | Reproducible, monitored |
| **TIER 7: Deployment** | 8h | 3-5× speedup (TensorRT), 1000+ QPS |
| **TIER 8: Fusion** | 6h | +2-3% accuracy |
| **TOTAL** | **70h** | **88-92%+ accuracy, production-ready** |

---

## 🚦 **GETTING STARTED**

### **Step 1: Clone and Setup**
```bash
cd /home/sina/projects/miner_b/stage1_pro_modular_training_system
git checkout -b stage1_finalV
cd stage1_finalV
```

### **Step 2: Install Dependencies**
```bash
poetry install
# Or if using pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### **Step 3: Run Smoke Test**
```bash
# After Phase 0.4 is complete
python scripts/train.py --config configs/hydra/phase1.yaml --data.batch_size 8 --training.num_epochs 1
```

### **Step 4: Start Training**
```bash
# Full training (50 epochs)
python scripts/train.py --config configs/hydra/phase1.yaml
```

---

**START WITH TIER 0: FOUNDATION - 6h to get a working CLI!** 🚀

