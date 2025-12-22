# Project Structure Guide

## Directory Layout

### /phase0_testnet/ (317 MB)
**Purpose:** Testnet validation and testing phase

```
phase0_testnet/
├── streetvision-subnet/       # Official NATIX subnet repo (Git submodule)
│   ├── neurons/               # Miner/Validator implementations
│   ├── base_miner/            # Base miner logic
│   ├── natix/                 # NATIX-specific integration
│   ├── tests/                 # Unit tests
│   └── .gitignore             # Comprehensive ignore rules
├── venv/                      # [GITIGNORED] Python virtual environment
├── models/                    # [GITIGNORED] Testnet model cache
├── data/                      # [GITIGNORED] Testnet datasets
└── logs/                      # [GITIGNORED] Execution logs
```

**Status:** Active testnet code, maintained Git repository

---

### /streetvision_cascade/ (69 GB → 50 MB after cleanup)
**Purpose:** Production 6-stage cascade training and deployment

```
streetvision_cascade/
├── models/                    # [GITIGNORED - 60GB] Model files
│   ├── stage1_dinov3/         # DINOv3-ViT-H16+ (3.36GB)
│   ├── stage2_yolo/           # YOLOv11-X (110MB)
│   ├── stage2_rfdetr/         # RF-DETR-Medium (83MB)
│   ├── stage3_glm/            # GLM-4.6V-Flash (20GB)
│   ├── stage3_molmo/          # Molmo-2-8B (33GB)
│   ├── stage4_florence/       # Florence-2-Large (1.5GB)
│   ├── quantized/             # [Empty] AWQ-quantized models
│   └── tensorrt/              # [Empty] TensorRT engines
├── data/                      # [GITIGNORED - 328MB] Training data
│   ├── natix_official/        # Official NATIX 8K dataset
│   ├── hard_cases/            # Hard-case mining results
│   ├── synthetic_sdxl/        # SDXL-generated synthetics
│   └── validation/            # Validation splits
├── scripts/                   # ✅ COMMIT Utility scripts
│   ├── download_models.py     # Automated model downloads
│   ├── monitor_download_progress.py
│   └── train_stage1_head.py   # [TO ADD] Stage 1 training
├── configs/                   # ✅ COMMIT Configuration files
│   ├── stage1_dinov3.yaml
│   ├── cascade_thresholds.yaml
│   └── training_defaults.yaml
├── .venv/                     # [GITIGNORED - 8GB] Virtual environment
├── cache/                     # [GITIGNORED] Model cache
├── checkpoints/               # [GITIGNORED] Training checkpoints
├── logs/                      # [GITIGNORED] Training logs
├── requirements.txt           # ✅ COMMIT Python dependencies
├── SETUP_STATUS.md            # Status documentation
├── DOWNLOAD_STATUS.md         # Model download tracker
└── NEXT_STEPS.md              # Next actions
```

**Status:** Production codebase, needs .gitignore and cleanup

---

### /ROOT/ (49 files → 7 essential docs)

**KEEP (Essential Documentation):**
```
✅ README.md                          # [TO CREATE] Project overview
✅ START_HERE.md                      # Entry point guide
✅ START_HERE_REALISTIC.md            # Realistic expectations
✅ REALISTIC_DEPLOYMENT_PLAN.md       # Complete 5,036-line guide
✅ COMPLETE_DEPLOYMENT_PLAN.md        # Phase 0-5 roadmap
✅ COMPLETE_DEPLOYMENT_PLAN_PART2.md  # Scaling guide
✅ WHAT_CHANGED.md                    # Changelog
✅ PROJECT_STRUCTURE.md               # [TO CREATE] This file
```

**ARCHIVE (43 legacy files):**
```
archive/legacy_docs/
├── fd1.md through fd17.md      # Iteration history (17 files)
├── ff.md, ff1-ff15.md          # Feedback notes (16 files)
├── most.md through most6.md    # Analysis variants (7 files)
├── cursor_*.md                 # Validation notes (2 files)
└── [other legacy]              # Misc planning docs (1 file)
```

**DELETE (Binary - not suitable for Git):**
```
❌ yolo11x.pt (110MB)           # Use external download script
```

---

## File Type Categories

### Code (Commit to Git)
- Python scripts: `*.py`
- Configuration: `*.yaml`, `*.json`, `*.toml`
- Requirements: `requirements.txt`, `setup.py`, `pyproject.toml`
- Shell scripts: `*.sh`

**Size:** ~20 MB total

### Documentation (Commit to Git)
- Essential markdown: 7 files (~200 KB)
- Code documentation: `docs/` directories

### Models (External - NEVER commit)
- Model weights: `*.pt`, `*.pth`, `*.safetensors`, `*.bin`
- ONNX models: `*.onnx`
- TensorRT engines: `*.engine`

**Size:** 60.6 GB (manage via download scripts + HuggingFace)

### Data (External - NEVER commit)
- Training datasets: `data/` directories
- Validation sets: `validation/` directories
- Synthetic data: `synthetic_*/` directories

**Size:** 328 MB (document structure, use download scripts)

### Virtual Environments (Never commit)
- Python venvs: `venv/`, `.venv/`, `env/`

**Size:** 11+ GB (recreate via `pip install -r requirements.txt`)

---

## Git Management Strategy

### What Goes in GitHub
```
✅ Source code (~20 MB)
✅ Documentation (~200 KB)
✅ Configuration files (~100 KB)
✅ Requirements/dependencies (~10 KB)
✅ Scripts for downloading models/data
✅ Project structure documentation
```

### What Stays External
```
❌ Model files (60.6 GB) → HuggingFace + download scripts
❌ Training data (328 MB) → NATIX official + download instructions
❌ Virtual environments (11 GB) → Recreate via requirements.txt
❌ Logs/checkpoints → Local only, gitignored
❌ Cache directories → Local only, gitignored
```

### Total Size After Cleanup
- **Before:** 69.3 GB
- **After:** ~50 MB (GitHub-friendly ✅)

---

## External Dependencies

### Model Downloads
All models managed via `streetvision_cascade/scripts/download_models.py`:

1. DINOv3-vith16plus: `facebook/dinov3-vith16plus-pretrain-lvd1689m`
2. RF-DETR: `microsoft/rt-detr-medium`
3. YOLOv12-X: `ultralytics/yolov12x.pt`
4. GLM-4.6V: `z-ai/GLM-4.6V-Flash-9B`
5. Molmo-2-8B: `allenai/Molmo-2-8B`
6. Florence-2: `microsoft/Florence-2-large`

**Total Download:** ~60 GB (one-time, 30-60 min on fast connection)

### Dataset Access
- **NATIX Official Dataset:** Contact NATIX team for access
- **Expected Size:** 8,000 labeled images (~500 MB compressed)
- **Format:** JPG/PNG images + CSV labels

---

## Development Workflow

### Initial Setup
```bash
1. Clone repository
2. Create virtual environment
3. pip install -r requirements.txt
4. Run download_models.py
5. Download NATIX dataset
6. Start training Stage 1
```

### Daily Development
```bash
1. Activate virtual environment
2. Work in streetvision_cascade/
3. Models/data already downloaded (cached locally)
4. Train, validate, iterate
5. Commit code changes only (not models/data)
```

### Before Pushing to GitHub
```bash
1. Verify .gitignore catches all large files
2. Check repo size: should be ~50 MB
3. Run: git status (verify no *.pt, *.pth, */venv/*, */data/* files staged)
4. Commit and push
```

---

## Size Breakdown Table

| Directory | Before Cleanup | After Cleanup | Status |
|-----------|----------------|---------------|--------|
| **Code** | ~20 MB | ~20 MB | ✅ Commit |
| **Essential Docs** | 200 KB | 200 KB | ✅ Commit |
| **Legacy Docs** | 1.7 GB | 0 (archived) | ❌ Delete/Archive |
| **Models** | 60.6 GB | 0 (external) | ❌ .gitignore |
| **Data** | 328 MB | 0 (external) | ❌ .gitignore |
| **Virtual Envs** | 11 GB | 0 (recreate) | ❌ .gitignore |
| **Binary Files** | 110 MB | 0 (external) | ❌ .gitignore |
| **Configs/Scripts** | ~5 MB | ~5 MB | ✅ Commit |
| **TOTAL** | **69.3 GB** | **~50 MB** | **GitHub Ready** |

---

## Quick Reference Commands

### Check current repo size
```bash
du -sh miner_b/
du -sh miner_b/* | sort -h
```

### Verify .gitignore works
```bash
git status --ignored
git check-ignore -v <file>
```

### Find large files
```bash
find . -type f -size +100M
find . -name "*.pt" -o -name "*.pth" -o -name "*.safetensors"
```

### Clean up before commit
```bash
# Remove cached Python files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Verify no large files staged
git ls-files -s | awk '{if ($4 > 1000000) print $4, $2}'
```
