# SSH GPU Server Deployment Guide - Complete Preparation

**Goal**: Deploy production-ready NATIX Stage-1 pipeline to rented GPU server
**Status**: Step-by-step preparation guide
**Date**: 2025-12-30

---

## 🎯 Overview

You will:
1. Prepare everything locally (code + data)
2. Push code to GitHub
3. Rent GPU server (e.g., vast.ai, runpod.io)
4. SSH to server
5. Clone code + download data
6. Run full pipeline

**Time to complete**: ~2-3 hours (most time is data transfer)

---

## 📋 Pre-Flight Checklist (LOCAL)

### Step 1: Clean Up Local Outputs (Save Space)

**Current status**: `outputs/` folder is 23GB (old test runs)

```bash
cd /home/sina/projects/miner_b/stage1_ultimate

# OPTION A: Delete all old outputs (recommended - save space)
rm -rf outputs/*

# OPTION B: Keep latest run (if you want to preserve something)
# (manually inspect outputs/ and delete what you don't need)
```

**After cleanup**: Git won't track outputs/ (it's in .gitignore)

---

### Step 2: Verify Code is Production-Ready

```bash
cd /home/sina/projects/miner_b/stage1_ultimate

# Check all Python files compile
find src -name "*.py" -exec python3 -m py_compile {} \;

# Verify structure
tree -L 2 -I '__pycache__|*.pyc|.venv*'
```

**Expected output**:
```
.
├── configs/                  # Hydra configs
├── DAY5_COMPLETE.md         # Documentation
├── DAYS_1_TO_5_COMPLETE.md  # Documentation
├── docs/                    # Architecture docs
├── examples/                # Example scripts (optional)
├── pyproject.toml          # Dependencies
├── README.md               # Project overview
├── scripts/
│   ├── download_full_dataset.py
│   └── train_cli_v2.py     # MAIN PRODUCTION CLI
├── src/
│   ├── contracts/          # Artifact schema
│   ├── data/              # Data modules
│   ├── pipeline/          # DAG engine
│   ├── streetvision/      # Production package
│   └── training/          # Training modules
├── tests/                 # Integration tests
└── tools/                 # Utility scripts
```

---

### Step 3: Create .gitignore (If Not Exists)

```bash
cd /home/sina/projects/miner_b/stage1_ultimate

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.venv*/
venv*/

# Outputs (don't commit large files)
outputs/
runs/
logs/
*.pth
*.pt
*.ckpt
*.safetensors

# Hydra
.hydra/
multirun/

# Data (don't commit datasets)
data/
datasets/
*.tar.gz
*.zip

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary
*.tmp
*.log
*.txt
!requirements.txt
EOF
```

---

### Step 4: Initialize Git Repository (If Not Already)

```bash
cd /home/sina/projects/miner_b/stage1_ultimate

# Initialize git (if not already a repo)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Production-ready Stage-1 pipeline (Days 1-7 complete)

- Atomic IO with manifest-last commit
- Centralized metrics (no MCC drift)
- Resume logic (crash recovery)
- Integration tests (17 tests)
- Evaluation reports (Day 6)
- Profiling tools (Day 7)
- torch.compile support (Day 7)
- 3 critical bugs fixed in Phase-4
- ECE calibration metrics
- Full documentation
"
```

---

### Step 5: Push to GitHub

```bash
# Create GitHub repo (if not exists)
# Go to github.com → New Repository → "natix-stage1-ultimate"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/natix-stage1-ultimate.git

# Push code
git branch -M main
git push -u origin main
```

**IMPORTANT**: Make repo **PRIVATE** if code is proprietary!

---

## 📦 Data Preparation Strategy

You have 2 options for getting data on GPU server:

### **Option A: Download from HuggingFace (RECOMMENDED)**

**Pros**: Fast, no upload needed
**Cons**: Requires HuggingFace dataset to be public/accessible

**On GPU server later**:
```bash
# Use your download script
python scripts/download_full_dataset.py
```

### **Option B: Upload from Local**

**Pros**: Full control, works with custom data
**Cons**: Slow upload (depends on your internet)

**Prepare locally**:
```bash
# Create tarball of NATIX data
cd ~/data
tar -czf natix_dataset.tar.gz natix_subset/

# Size check
du -sh natix_dataset.tar.gz
```

**Upload to server later** (after renting):
```bash
scp natix_dataset.tar.gz user@server:/workspace/data/
```

---

## 🖥️ GPU Server Requirements

### Minimum Specs (For Full Pipeline)

- **GPUs**: 2× A6000 (48GB VRAM each) **OR** 2× RTX 4090 (24GB each)
- **RAM**: 128GB system RAM
- **Storage**: 500GB SSD (for data + checkpoints)
- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.1+
- **PyTorch**: 2.5+ (with CUDA support)

### Recommended Providers

1. **vast.ai** (cheapest)
   - Filter: 2× A6000, 128GB RAM, 500GB storage
   - ~$1-2/hour
   - Spot instances (can be interrupted)

2. **runpod.io** (reliable)
   - Community Cloud: ~$1.50-2.50/hour
   - Secure Cloud: ~$3-4/hour (guaranteed)

3. **Lambda Labs** (premium)
   - 2× A6000: ~$3/hour
   - Very stable

---

## 🚀 Deployment Steps (ON GPU SERVER)

### Step 1: Rent GPU Server

**On vast.ai**:
1. Go to https://vast.ai
2. Filter: "2x A6000" OR "2x RTX 4090"
3. Filter: RAM >= 128GB
4. Filter: Disk >= 500GB
5. Sort by: "$/hour" (cheapest first)
6. Click "Rent" → Use "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel" image
7. **IMPORTANT**: Enable SSH access!

**Get SSH command**:
```
ssh -p XXXXX root@XX.XXX.XX.XX -L 8080:localhost:8080
```

### Step 2: SSH to Server

```bash
# From your local machine
ssh -p XXXXX root@XX.XXX.XX.XX

# You're now on the GPU server!
```

---

### Step 3: Install System Dependencies (ON SERVER)

```bash
# Update system
apt-get update
apt-get install -y git wget curl vim htop tree

# Verify GPUs
nvidia-smi

# Should show 2× GPUs with ~48GB VRAM each
```

---

### Step 4: Clone Your Code (ON SERVER)

```bash
cd /workspace

# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/natix-stage1-ultimate.git
cd natix-stage1-ultimate

# Verify structure
tree -L 2 -I '__pycache__|*.pyc'
```

---

### Step 5: Create Virtual Environment (ON SERVER)

```bash
cd /workspace/natix-stage1-ultimate

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

### Step 6: Install Dependencies (ON SERVER)

```bash
# Install PyTorch with CUDA (if not in base image)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -e .

# Or install from requirements if you created one:
# pip install -r requirements.txt

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Expected output:
# PyTorch: 2.5.1+cu124
# CUDA: True
# GPUs: 2
```

---

### Step 7: Get Dataset (ON SERVER)

**Option A: Download from HuggingFace**:
```bash
cd /workspace/natix-stage1-ultimate

# Run your download script
python scripts/download_full_dataset.py

# Check data
ls -lh ~/data/natix_subset/
```

**Option B: Upload from Local**:
```bash
# On LOCAL machine:
scp -P XXXXX ~/data/natix_dataset.tar.gz root@XX.XXX.XX.XX:/workspace/data/

# On SERVER:
cd /workspace/data
tar -xzf natix_dataset.tar.gz
ls -lh natix_subset/
```

---

### Step 8: Update Config Paths (ON SERVER)

```bash
cd /workspace/natix-stage1-ultimate

# Edit config to point to data location
vim configs/config.yaml

# Update this line:
data:
  data_root: /workspace/data/natix_subset  # ← Update path!
```

---

### Step 9: Test Small Run (ON SERVER)

```bash
cd /workspace/natix-stage1-ultimate
source .venv/bin/activate

# Run Phase-1 with tiny config (quick test)
python scripts/train_cli_v2.py \
    pipeline.phases=[phase1] \
    data.max_samples=100 \
    training.epochs=1 \
    hardware.num_gpus=2

# Should complete in ~5-10 minutes
# Check outputs/
ls -lh outputs/stage1_ultimate/runs/
```

---

### Step 10: Run Full Pipeline (ON SERVER)

```bash
cd /workspace/natix-stage1-ultimate
source .venv/bin/activate

# Run full pipeline: Phase-1 → Phase-2 → Phase-4 → Phase-5 → Phase-6
python scripts/train_cli_v2.py \
    pipeline.phases=[phase1,phase2,phase4,phase5,phase6] \
    hardware.num_gpus=2 \
    data.data_root=/workspace/data/natix_subset

# This will run for ~6-12 hours (depends on dataset size)
```

---

### Step 11: Monitor Progress (ON SERVER)

**In another SSH session**:
```bash
ssh -p XXXXX root@XX.XXX.XX.XX

# Watch GPU usage
watch -n 1 nvidia-smi

# Watch logs
tail -f outputs/stage1_ultimate/runs/*/logs/*.log

# Check manifests (phase completion)
find outputs -name "manifest.json" -exec echo {} \; -exec cat {} \;
```

---

### Step 12: Download Results (FROM LOCAL)

```bash
# After pipeline completes, download bundle
scp -P XXXXX root@XX.XXX.XX.XX:/workspace/natix-stage1-ultimate/outputs/stage1_ultimate/runs/LATEST/phase6_bundle/export/bundle.json ./

# Download checkpoints
scp -r -P XXXXX root@XX.XXX.XX.XX:/workspace/natix-stage1-ultimate/outputs/stage1_ultimate/runs/LATEST/ ./results/
```

---

## 🔍 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python scripts/train_cli_v2.py \
    pipeline.phases=[phase1] \
    data.dataloader.batch_size=16 \
    data.dataloader.val_batch_size=16 \
    hardware.num_gpus=2
```

### Issue: Data Not Found

**Solution**: Check paths
```bash
# Verify data location
ls -lh /workspace/data/natix_subset/

# Update config
vim configs/config.yaml
```

### Issue: Import Errors

**Solution**: Reinstall package
```bash
source .venv/bin/activate
pip install -e . --force-reinstall
```

### Issue: Pipeline Crashes

**Solution**: Use resume logic (Day 5)
```bash
# Just re-run same command
# Resume logic will skip completed phases automatically
python scripts/train_cli_v2.py pipeline.phases=[phase1,phase2,phase4,phase5,phase6]
```

---

## 📊 Expected Timeline

| Phase | Time | Output |
|-------|------|--------|
| Phase-1 (Baseline) | 2-3 hours | model_best.pth, val_calib_logits.pt |
| Phase-2 (Threshold) | 5-10 min | best_threshold.json |
| Phase-4 (ExPLoRA) | 3-4 hours | explora_merged.pth |
| Phase-5 (SCRC) | 10-15 min | calibration_params.json |
| Phase-6 (Bundle) | 2-3 min | bundle.json |
| **Total** | **~6-8 hours** | **Deployment-ready bundle** |

---

## 💰 Cost Estimate

**Using vast.ai (2× A6000)**:
- Rate: ~$1.50/hour
- Full pipeline: ~8 hours
- **Total cost**: ~$12-15

**Using runpod.io (2× A6000)**:
- Rate: ~$2.50/hour
- Full pipeline: ~8 hours
- **Total cost**: ~$20-25

---

## ✅ Final Checklist

**LOCAL (Before Deployment)**:
- [ ] Clean up outputs/ folder
- [ ] Verify all Python files compile
- [ ] Create .gitignore
- [ ] Push code to GitHub
- [ ] Prepare data strategy (HuggingFace or local)

**SERVER (After Renting)**:
- [ ] SSH to server
- [ ] Clone code from GitHub
- [ ] Install dependencies
- [ ] Get dataset
- [ ] Update config paths
- [ ] Test small run (100 samples)
- [ ] Run full pipeline
- [ ] Monitor progress
- [ ] Download results

---

## 🎯 What to Change in Code (NONE!)

**GOOD NEWS**: No code changes needed for SSH deployment!

The production code is already:
- ✅ Path-agnostic (uses Hydra config)
- ✅ GPU-aware (auto-detects num_gpus)
- ✅ Crash-safe (resume logic)
- ✅ Deployment-ready (bundle export)

**Only change**: Config file paths (update `data_root` in configs/config.yaml)

---

## 🚀 Quick Deployment Script (Copy-Paste)

Save this as `deploy.sh` on SERVER:

```bash
#!/bin/bash
set -e

# Setup
cd /workspace
git clone https://github.com/YOUR_USERNAME/natix-stage1-ultimate.git
cd natix-stage1-ultimate

# Python env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Get data (choose one):
# Option A: Download
# python scripts/download_full_dataset.py

# Option B: Extract uploaded tarball
# cd /workspace/data && tar -xzf natix_dataset.tar.gz

# Update config
sed -i 's|data_root:.*|data_root: /workspace/data/natix_subset|' configs/config.yaml

# Test run
python scripts/train_cli_v2.py \
    pipeline.phases=[phase1] \
    data.max_samples=100 \
    training.epochs=1 \
    hardware.num_gpus=2

echo "✅ Test complete! Now run full pipeline manually."
```

**Usage**:
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 📞 Support

If you encounter issues:
1. Check logs: `tail -f outputs/*/logs/*.log`
2. Check manifests: `cat outputs/*/manifest.json`
3. Check GPU memory: `nvidia-smi`
4. Check integration tests: `pytest tests/integration/ -v`

---

**Status**: Ready for deployment! 🚀

**Next step**: Clean up local outputs/, push to GitHub, rent GPU, and deploy!
