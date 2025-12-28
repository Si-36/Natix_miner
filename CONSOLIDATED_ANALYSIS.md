# CONSOLIDATED PLAN ANALYSIS
## Comparing olanzapin.md vs ULTIMATE_120_TODO_PLAN.md

Generated: December 28, 2025

---

## EXECUTIVE SUMMARY

**Both plans are now remarkably similar** - they target the same goals:
- Production-grade Stage-1 Pro Modular Training System
- 88-92%+ validation accuracy (from 69%)
- 2-3× faster training/inference
- SOTA 2025 features (ExPLoRA, DoRAN, Flash Attention 3, LCRON, Gatekeeper)
- Complete MLOps infrastructure

**KEY FINDING**: `ULTIMATE_120_TODO_PLAN.md` is the **superset** - it contains nearly everything from `olanzapin.md` PLUS:
1. **DAG Pipeline Architecture** (not in olanzapin)
2. **Split Contracts enforced as code** (not in olanzapin)
3. **Artifact Registry** (not in olanzapin)
4. **DAGEngine with automatic dependency resolution** (not in olanzapin)
5. **More detailed 120 TODOs with full code implementations**

**RECOMMENDATION**: Use `ULTIMATE_120_TODO_PLAN.md` as the master plan, as it's more comprehensive and better organized.

---

## STRUCTURAL COMPARISON

| Aspect | olanzapin.md | ULTIMATE_120_TODO_PLAN.md | Winner |
|--------|--------------|---------------------------|---------|
| Organization | 10 Phases with 120 TODOs | 8 Tiers with 120 TODOs | **Tie** (both have 120 TODOs) |
| Time Estimate | 90 hours | 150 hours | **olanzapin** (more aggressive) |
| Code Detail | High (code snippets) | **VERY HIGH** (complete implementations) | **ULTIMATE** |
| Foundation | Lightning + Hydra | **DAG Pipeline + Lightning + Hydra** | **ULTIMATE** |
| Architecture | Monolithic pipeline | **DAG orchestration with dependencies** | **ULTIMATE** |
| Testing | Unit + integration | **Unit + integration + acceptance** | **Tie** |
| Documentation | 10 docs (3000 lines) | **7 docs (less detail)** | **olanzapin** |
| Data Leakage Prevention | Manual checks | **Split contracts enforced as code** | **ULTIMATE** |
| Artifact Management | Manual paths | **Artifact Registry (single source of truth)** | **ULTIMATE** |

---

## FEATURE COMPARISON

### Machine Learning Features

| Feature | olanzapin.md | ULTIMATE_120_TODO_PLAN.md | Notes |
|---------|--------------|---------------------------|-------|
| **ExPLoRA** | ✅ TODO 43 | ✅ TODO 11 (+8.2% accuracy) | **Both** |
| **DoRAN** | ✅ TODO 45 | ✅ TODO 12 (+1-3% over LoRA) | **Both** |
| **Multi-View Inference** | ✅ TODO 11-14 | ✅ TODO 14 (1 global + 3×3 tiles) | **Both** |
| **Flash Attention 3** | ❌ Missing | ✅ TODO 13 (1.5-2× speed) | **ULTIMATE only** |
| **LCRON Loss** | ✅ TODO 39 | ✅ TODO 19 (+3-5% cascade) | **Both** |
| **Gatekeeper Loss** | ✅ TODO 40 | ✅ TODO 20 (+2-3% calibration) | **Both** |
| **Koleo Loss** | ✅ TODO 34 | ✅ TODO 18 | **Both** |
| **SAM Optimizer** | ✅ TODO 35 | ✅ TODO 21 (+1-2% accuracy) | **Both** |
| **Curriculum Learning** | ✅ TODO 36 | ✅ TODO 26 | **Both** |
| **MixUp/CutMix** | ✅ TODO 38 | ✅ TODO 27 | **Both** |
| **AugMax** | ✅ TODO 38 | ✅ TODO 28 | **Both** |
| **Hard Negative Mining** | ✅ TODO 82 | ✅ TODO 18 (orange objects) | **Both** |
| **Domain Adaptation** | ✅ TODO 83 | ✅ TODO 68 | **Both** |
| **Class Balancing** | ✅ TODO 86 | ✅ TODO 67 (50/50 NATIX/Mapillary) | **ULTIMATE** |
| **MC Dropout** | ✅ TODO 89 | ❌ Missing | **olanzapin only** |
| **Evidential Deep Learning** | ❌ Missing | ✅ TODO 17 (7D uncertainty) | **ULTIMATE only** |
| **Hierarchical Stochastic Attention** | ❌ Missing | ✅ TODO 17 | **ULTIMATE only** |

### Calibration Features

| Feature | olanzapin.md | ULTIMATE_120_TODO_PLAN.md | Notes |
|---------|--------------|---------------------------|-------|
| **Temperature Scaling** | ✅ TODO 51 | ✅ TODO 31 | **Both** |
| **Beta Calibration** | ✅ TODO 52 | ✅ TODO 32 | **Both** |
| **Class-wise Calibration** | ✅ TODO 53 | ✅ TODO 33 | **Both** |
| **Ensemble Calibration** | ✅ TODO 53 | ✅ TODO 35 | **Both** |
| **Conformal Prediction** | ✅ TODO 54 | ✅ TODO 36 | **Both** |
| **Isotonic Regression** | ✅ TODO 56 | ✅ TODO 34 | **Both** |
| **Reliability Diagrams** | ✅ TODO 57 | ✅ TODO 34 | **Both** |
| **Slice-wise Calibration** | ✅ TODO 58 | ❌ Missing | **olanzapin only** |
| **Calibration Drift Detection** | ✅ TODO 59 | ✅ TODO 36 | **Both** |
| **SCRC/CRCP** | ✅ Fixed in TODO 6 | ✅ TODO 55 (enhanced) | **Both** |

### Optimization Features

| Feature | olanzapin.md | ULTIMATE_120_TODO_PLAN.md | Notes |
|---------|--------------|---------------------------|-------|
| **torch.compile** | ✅ | ✅ TODO 29 (30-50% free speedup) | **Both** |
| **Flash Attention 3** | ❌ Missing | ✅ TODO 13 (1.5-2× speed) | **ULTIMATE only** |
| **FSDP2** | ✅ | ✅ TODO 30 (2× memory reduction) | **Both** |
| **ONNX Export** | ✅ TODO 71 | ✅ TODO 51 | **Both** |
| **TensorRT** | ✅ TODO 72 | ✅ TODO 52 (3-5× speed) | **Both** |
| **Quantization (FP16/INT8)** | ✅ TODO 72 | ✅ TODO 53 | **Both** |
| **Distillation** | ✅ TODO 73 | ✅ TODO 53 | **Both** |

### MLOps Features

| Feature | olanzapin.md | ULTIMATE_120_TODO_PLAN.md | Notes |
|---------|--------------|---------------------------|-------|
| **MLflow Tracking** | ✅ TODO 77 | ✅ TODO 76 | **Both** |
| **DVC Versioning** | ✅ TODO 81 | ✅ TODO 77 | **Both** |
| **CI/CD Pipeline** | ✅ TODO 115 | ✅ TODO 78 | **Both** |
| **Prometheus Monitoring** | ✅ TODO 78 | ✅ TODO 80 | **Both** |
| **Grafana Dashboards** | ✅ TODO 81 | ✅ TODO 81 | **Both** |
| **Feature Store (Feast)** | ✅ TODO 79 | ✅ TODO 79 | **Both** |
| **Automated Retraining** | ✅ TODO 82 | ✅ TODO 82 | **Both** |
| **A/B Testing** | ✅ TODO 79 | ✅ TODO 58 | **Both** |
| **Shadow Deployment** | ✅ TODO 80 | ✅ TODO 57 | **Both** |

### Unique Features in ULTIMATE (NOT in olanzapin)

1. **DAG Pipeline Architecture** (TIER 0, TODOs 2-10)
   - `contracts/artifact_schema.py` - Artifact Registry
   - `contracts/split_contracts.py` - Split Contracts (enforced as code)
   - `contracts/validators.py` - Hard Validators
   - `pipeline/phase_spec.py` - Phase Specifications
   - `pipeline/dag_engine.py` - DAGEngine with dependency resolution
   - **Benefit**: Zero data leakage, fail-fast validation, clear phase contracts

2. **Split Contracts Enforced as Code** (TODO 3)
   - `SplitPolicy` class with validation methods
   - Cannot violate split usage rules - system fails immediately
   - **Benefit**: Correctness by construction

3. **Artifact Registry** (TODO 2)
   - `ArtifactSchema` class with all paths as properties
   - Single source of truth for all file paths
   - **Benefit**: Zero hardcoded paths, cannot "forget to save X"

4. **Hard Validators** (TODO 4)
   - `ArtifactValidator` class with check methods
   - Fail immediately if artifacts missing/corrupted
   - **Benefit**: Catch errors early, clear error messages

5. **DAGEngine** (TODO 10)
   - Automatic dependency resolution
   - `resolve_dependencies(phase)` method
   - **Benefit**: No manual orchestration, clear phase ordering

6. **ExPLoRA Enhanced** (TODO 11)
   - More detailed implementation than olanzapin
   - Includes noise injection stabilization
   - Expected +8.2% accuracy

7. **DoRAN Enhanced** (TODO 12)
   - DoRA + Noise stabilization + Auxiliary network
   - More stable than standard DoRA
   - Expected +1-3% over LoRA/DoRA

8. **Flash Attention 3 Integration** (TODO 13)
   - 1.5-2× speedup over Flash Attention 2
   - Complete implementation
   - **Not in olanzapin**

9. **7D Uncertainty Features** (TODO 17)
   - Evidential Deep Learning (Dirichlet)
   - Hierarchical Stochastic Attention
   - Max_prob, variance, entropy, max_minus_mean, crop_disagreement, epistemic, aleatoric
   - **More detailed than olanzapin's 5D features**

10. **Complete 120 TODOs with Full Code** (All TODOs)
    - Every TODO has complete, copy-paste ready code
    - No stubs, no placeholders
    - **More detailed than olanzapin**

### Unique Features in olanzapin (NOT in ULTIMATE)

1. **MC Dropout** (TODO 89)
   - Monte Carlo Dropout for uncertainty ensembling
   - Cheap ensembling method
   - **Not in ULTIMATE**

2. **Slice-wise Calibration** (TODO 58)
   - Calibrate per slice (day/night/weather/camera ID)
   - Better calibration on heterogeneous data
   - **Not in ULTIMATE**

3. **More Comprehensive Documentation** (TODOs 101-110)
   - 10 documentation files (3000+ lines)
   - ARCHITECTURE.md, API.md, TRAINING_GUIDE.md, DEPLOYMENT.md, RESEARCH_NOTES.md, MLOPS_GUIDE.md, CONFIG_REFERENCE.md, DEBUGGING_GUIDE.md, MONITORING_GUIDE.md, FAQ.md
   - **More detailed than ULTIMATE's 7 docs**

4. **Complete Testing Infrastructure** (TODOs 91-100)
   - Unit tests (5 test files, 1000 lines)
   - Integration tests (5 test files, 800 lines)
   - Acceptance tests (1 test file, 400 lines)
   - **More detailed than ULTIMATE**

5. **More Detailed Setup Files** (TODOs 111-115)
   - setup.py with entry points
   - Makefile with build automation
   - .gitignore with comprehensive rules
   - CI/CD pipeline with GitHub Actions
   - **More complete than ULTIMATE**

---

## ARCHITECTURAL COMPARISON

### olanzapin.md Architecture

```
stage1_pro_modular_training_system/
├── src/
│   ├── core/                    # Pipeline orchestration
│   │   ├── pipeline.py          # Main orchestrator
│   │   ├── components.py        # Component factory
│   │   ├── contracts.py         # Split contracts
│   │   ├── artifact_registry.py # Artifact registry
│   │   ├── config_validator.py  # Pydantic validation
│   │   └── mlflow_tracker.py    # MLflow integration
│   ├── models/
│   │   ├── backbone.py
│   │   ├── multi_view.py
│   │   ├── failure_gate.py
│   │   ├── uncertainty.py
│   │   ├── cascade_router.py
│   │   └── explora.py
│   ├── training/
│   │   ├── lightning_module.py # Lightning trainer
│   │   ├── explora.py
│   │   ├── doran.py
│   │   ├── koleo_loss.py
│   │   ├── sam_optimizer.py
│   │   ├── lcron_loss.py
│   │   └── gatekeeper_loss.py
│   ├── calibration/
│   ├── evaluation/
│   ├── data/
│   └── deployment/
├── configs/                    # Hydra configs
├── tests/                      # Unit/integration/acceptance
└── docs/                       # 10 documentation files
```

**Strengths**:
- ✅ Clean separation of concerns
- ✅ PyTorch Lightning integration
- ✅ Hydra + Pydantic configs
- ✅ Comprehensive documentation (10 files, 3000+ lines)
- ✅ Complete testing infrastructure (100% coverage target)

**Weaknesses**:
- ❌ No DAG orchestration
- ❌ No automatic dependency resolution
- ❌ Split contracts not enforced as code
- ❌ Artifact management is manual

### ULTIMATE_120_TODO_PLAN.md Architecture

```
stage1_pro_system/
├── contracts/                  # ⭐ UNIQUE - DAG-based contracts
│   ├── artifact_schema.py     # Artifact Registry (single source of truth)
│   ├── split_contracts.py     # Split contracts (enforced as code)
│   └── validators.py          # Hard validators (fail-fast)
├── pipeline/                   # ⭐ UNIQUE - DAG orchestration
│   ├── phase_spec.py          # Phase specifications
│   ├── dag_engine.py          # DAGEngine (auto dependency resolution)
│   └── pipeline.py            # Main pipeline orchestrator
├── models/
│   ├── backbone.py
│   ├── explora.py             # ExPLoRA implementation
│   ├── doran_head.py          # DoRAN implementation
│   ├── flash_attn3.py         # Flash Attention 3
│   ├── multi_view.py
│   └── gatekeeper.py          # Gatekeeper confidence calibration
├── training/
│   ├── lightning_module.py
│   ├── lcron_loss.py
│   ├── gatekeeper_loss.py
│   ├── sam_optimizer.py
│   └── bilevel.py            # Bi-level optimization
├── calibration/
├── evaluation/
├── data/
│   ├── mapillary.py           # ⭐ UNIQUE - Mapillary Vistas
│   └── balanced_dataset.py   # ⭐ UNIQUE - 50/50 NATIX/Mapillary
├── deployment/
│   ├── onnx_export.py
│   ├── tensorrt.py            # ⭐ UNIQUE - TensorRT optimization
│   └── k8s/                  # ⭐ UNIQUE - Kubernetes deployment
├── configs/                   # Hydra configs
├── tests/
└── docs/                      # 7 documentation files
```

**Strengths**:
- ✅ **DAG orchestration** (unique)
- ✅ **Split contracts enforced as code** (unique)
- ✅ **Artifact Registry** (unique)
- ✅ **Automatic dependency resolution** (unique)
- ✅ **Fail-fast validation** (unique)
- ✅ Flash Attention 3 integration
- ✅ TensorRT optimization (3-5× speed)
- ✅ Mapillary Vistas integration
- ✅ Complete 120 TODOs with full code

**Weaknesses**:
- ❌ Less detailed documentation (7 files vs 10)
- ❌ Less detailed testing infrastructure
- ❌ Less detailed setup files

---

## RECOMMENDATION

### Use ULTIMATE_120_TODO_PLAN.md as the MASTER PLAN

**Reasons**:
1. **More complete architecture** - DAG orchestration is production-ready
2. **Better correctness** - Split contracts enforced as code
3. **Fail-fast validation** - Catch errors immediately
4. **More detailed TODOs** - Every TODO has complete code
5. **Unique high-value features** - Flash Attention 3, TensorRT, Mapillary integration

### Supplement with olanzapin.md's Best Parts

**Add to ULTIMATE from olanzapin**:
1. **MC Dropout** (olanzapin TODO 89) - Cheap ensembling for uncertainty
2. **Slice-wise Calibration** (olanzapin TODO 58) - Better calibration on heterogeneous data
3. **Comprehensive Documentation** (olanzapin TODOs 101-110) - 10 docs, 3000+ lines
4. **Complete Testing Infrastructure** (olanzapin TODOs 91-100) - Unit + integration + acceptance tests
5. **Setup Files** (olanzapin TODOs 111-115) - setup.py, Makefile, .gitignore, CI/CD

### Final Consolidated Plan

**Structure**:
- **TIER 0** (14h): DAG Pipeline + Contracts + Validators (from ULTIMATE)
- **TIER 1** (28h): ExPLoRA + DoRAN + Flash Attention 3 + Multi-view (from ULTIMATE)
- **TIER 2** (24h): Calibration + Evaluation (both plans)
- **TIER 3** (15h): Deployment + TensorRT (from ULTIMATE)
- **TIER 4** (10h): Multi-dataset Fusion + Mapillary (from ULTIMATE)
- **TIER 5** (20h): MLOps (both plans)
- **TIER 6** (10h): Testing (from olanzapin - more detailed)
- **TIER 7** (8h): Documentation (from olanzapin - more detailed)
- **TIER 8** (5h): Final Validation (both plans)

**Total Time**: 134 hours (~17 days)
- ULTIMATE: 150 hours
- olanzapin: 90 hours
- **Consolidated**: 134 hours (best of both)

---

## NEXT STEPS

1. **Copy ULTIMATE_120_TODO_PLAN.md structure** to `olanzapin.md`
2. **Add missing features** from olanzapin to ULTIMATE:
   - MC Dropout
   - Slice-wise Calibration
   - Comprehensive Documentation
   - Complete Testing Infrastructure
   - Setup Files
3. **Organize into 8 Tiers** (matching ULTIMATE)
4. **Update TODO numbers** to 120 total
5. **Verify no overlaps** or missing features
6. **Finalize consolidated plan**

---

## CONCLUSION

**The ULTIMATE_120_TODO_PLAN.md is the better foundation** - it has a more robust architecture (DAG orchestration), better correctness guarantees (split contracts), and more detailed implementations.

**The olanzapin.md has better depth** - more comprehensive documentation, testing infrastructure, and setup files.

**The best approach is to combine them**:
- Use ULTIMATE's DAG architecture as foundation
- Add olanzapin's documentation and testing depth
- Result: Production-ready, well-documented, thoroughly tested system

**Next Action**: Create consolidated plan document merging both plans.

