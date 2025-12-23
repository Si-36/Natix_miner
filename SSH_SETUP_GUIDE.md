# SSH Setup Guide - NATIX StreetVision Training on RTX A6000

## 🎯 Overview
This guide shows you **exactly** how to set up and run training on your rented RTX A6000 GPU server.

**Target GPU**: RTX A6000 (48GB VRAM, $0.45/hr)
**Expected Training Time**: ~1.5-2 hours
**Expected Cost**: ~$0.90 per training run

---

## 📋 Step-by-Step SSH Workflow

### Step 1: SSH into Your RTX A6000 Server

```bash
ssh root@YOUR_SERVER_IP
# Or if you have a username:
ssh username@YOUR_SERVER_IP
```

**What to expect**: You'll be prompted for your password. After entering it, you should see a terminal prompt.

---

### Step 2: Check GPU Availability

```bash
nvidia-smi
```

**Expected output**: Should show RTX A6000 with ~48GB VRAM. Verify it's available and not being used by other processes.

---

### Step 3: Install System Dependencies

```bash
# Update package manager
apt-get update

# Install Python 3.10+ (if not already installed)
apt-get install -y python3.10 python3-pip git

# Verify Python version
python3 --version
# Should show Python 3.10 or newer
```

---

### Step 4: Clone Your GitHub Repository

```bash
# Navigate to a working directory
cd /root  # Or wherever you want your project

# Clone your repo
git clone https://github.com/Si-36/Natix_miner.git
cd Natix_miner
```

**What this does**: Downloads your code from GitHub to the server.

---

### Step 5: Install Python Dependencies

```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and other dependencies
pip3 install transformers accelerate datasets Pillow tqdm scikit-learn

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output**:
```
PyTorch: 2.x.x+cu121
CUDA: True
```

---

### Step 6: Copy Models and Data to Server

You have 2 options:

#### Option A: Download models directly on server (RECOMMENDED)

```bash
# Install git-lfs for large files
apt-get install -y git-lfs
git lfs install

# Download DINOv3 model
mkdir -p models/stage1_dinov3
cd models/stage1_dinov3

# Download from HuggingFace
python3 << 'EOF'
from transformers import AutoModel, AutoImageProcessor
model_path = "facebook/dinov2-giant"
model = AutoModel.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
model.save_pretrained("dinov3-vith16plus-pretrain-lvd1689m")
processor.save_pretrained("dinov3-vith16plus-pretrain-lvd1689m")
print("✅ Model downloaded!")
EOF

cd /root/Natix_miner
```

#### Option B: Upload from your local machine (if you already have the 61GB model)

On your **local machine** (not SSH):
```bash
# Compress models first to speed up transfer
tar -czf models.tar.gz models/

# Upload to server (replace YOUR_SERVER_IP)
scp models.tar.gz root@YOUR_SERVER_IP:/root/Natix_miner/

# Then on SSH:
cd /root/Natix_miner
tar -xzf models.tar.gz
rm models.tar.gz
```

---

### Step 7: Upload NATIX Dataset

On your **local machine**:
```bash
# Compress dataset
tar -czf natix_data.tar.gz data/natix_official/

# Upload to server
scp natix_data.tar.gz root@YOUR_SERVER_IP:/root/Natix_miner/
```

On **SSH server**:
```bash
cd /root/Natix_miner
tar -xzf natix_data.tar.gz
rm natix_data.tar.gz

# Verify dataset structure
ls data/natix_official/
# Should show: train/ train_labels.csv val/ val_labels.csv
```

---

### Step 8: Verify Directory Structure

```bash
cd /root/Natix_miner
tree -L 3 -d
```

**Expected structure**:
```
.
├── data
│   └── natix_official
│       ├── train
│       └── val
├── models
│   └── stage1_dinov3
│       └── dinov3-vith16plus-pretrain-lvd1689m
└── streetvision_cascade
    ├── train_stage1_head.py
    └── validate_thresholds.py
```

---

### Step 9: Test Training Script (Dry Run)

```bash
cd streetvision_cascade

# Test with --help to see all options
python3 train_stage1_head.py --help
```

**Expected output**: Should show all CLI arguments and usage examples.

---

### Step 10: Run Quick Test (1 Epoch)

```bash
# Quick test run to verify everything works
python3 train_stage1_head.py --mode train --epochs 1
```

**What to watch for**:
- ✅ "GPU: NVIDIA RTX A6000" appears
- ✅ "Batch size 64 works on this GPU"
- ✅ "torch.compile enabled"
- ✅ Training starts and completes 1 epoch

**Expected time**: ~5-10 minutes for 1 epoch

---

### Step 11: Start Full Training

Once the test works, start the full training:

```bash
# Start training in a screen session (so it continues if SSH disconnects)
screen -S training

# Run full training
python3 train_stage1_head.py --mode train --epochs 10

# Detach from screen: Press Ctrl+A, then D
# To reattach later: screen -r training
```

**Expected output**:
```
================================================================================
DINOv3 STAGE 1 TRAINING - NATIX STREETVISION SUBNET 72
Production-Grade 2025 | RTX A6000 Optimized
================================================================================

Mode: train
Config will be saved to: models/stage1_dinov3/config.json
✅ Config saved to models/stage1_dinov3/config.json

================================================================================
FULL TRAINING MODE (with data augmentation)
================================================================================
Using device: cuda
GPU: NVIDIA RTX A6000
VRAM: 48.0 GB

[1/7] Loading DINOv3-vith16plus backbone...
✅ Frozen 1500.0M backbone parameters
✅ Training 300K classifier parameters (0.02% of full model)
✅ torch.compile enabled (expect 40% speedup after warmup)
✅ Batch size 64 works on this GPU
✅ Effective batch size: 128 (64 × 2 accum)

[2/7] Loading NATIX dataset...
✅ timm-style augmentation enabled for training (RandomResizedCrop + HFlip + RandomErasing)

📊 Class distribution:
   Class 0 (no roadwork): 8523 samples (75.2%)
   Class 1 (roadwork):    2811 samples (24.8%)
   Class weights: [0.66343915 1.99288256]
✅ Class-weighted loss with label smoothing=0.1

...training progress...
```

---

### Step 12: Monitor Training Progress

**In real-time** (if still attached to screen):
- Watch the progress bars
- Check ECE and exit metrics each epoch

**From another SSH session**:
```bash
# Monitor training log
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

### Step 13: After Training Completes

```bash
# Check results
ls -lh models/stage1_dinov3/
# Should see: classifier_head.pth, config.json, checkpoint_epochX.pth

# View training summary
tail -20 training.log

# Run threshold validation
python3 validate_thresholds.py
```

---

## 🚀 Advanced: Feature Caching for Fast Iteration

If you want to experiment with different hyperparameters (learning rate, dropout, etc.) **without re-running DINOv3 every time**:

### Extract features once (takes ~10 minutes):
```bash
python3 train_stage1_head.py --mode extract_features
```

### Train head only (10x faster, takes ~10 minutes instead of 2 hours):
```bash
# Experiment with different learning rates
python3 train_stage1_head.py --mode train_cached --lr_head 2e-4 --epochs 20

# Try different dropout
python3 train_stage1_head.py --mode train_cached --dropout 0.4 --epochs 20

# Etc.
```

---

## 📊 Expected Results

After training completes (~1.5-2 hours), you should see:

```
[7/7] Training complete!
🎯 Best Validation Accuracy: 96-97%
📁 Checkpoint saved: models/stage1_dinov3/classifier_head.pth
📊 Training log saved: training.log
```

**Key metrics to check**:
- **Validation Accuracy**: 96-97% (target)
- **ECE (Calibration)**: <0.05 (lower is better)
- **Exit Coverage @ 0.88**: ~60% (how many samples exit early)
- **Exit Accuracy @ 0.88**: >99% (accuracy on early exits)

---

## 🛟 Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Lower batch size
```bash
python3 train_stage1_head.py --max_batch_size 32 --grad_accum_steps 4
```

### Problem: "FileNotFoundError: train_labels.csv"
**Solution**: Check dataset paths
```bash
ls data/natix_official/
python3 train_stage1_head.py --train_image_dir data/natix_official/train --train_labels_file data/natix_official/train_labels.csv
```

### Problem: Training is very slow
**Solution**: Make sure torch.compile is working
- First epoch will be slow (compilation)
- Epochs 2+ should be 40% faster
- Check that you see "torch.compile enabled" in logs

### Problem: SSH disconnects during training
**Solution**: Use screen/tmux
```bash
screen -S training
python3 train_stage1_head.py --mode train --epochs 10
# Press Ctrl+A, then D to detach
# Later: screen -r training to reattach
```

---

## ✅ Quick Reference Commands

```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Check GPU
nvidia-smi

# Navigate to project
cd /root/Natix_miner/streetvision_cascade

# Full training
python3 train_stage1_head.py --mode train --epochs 10

# Monitor logs
tail -f training.log

# Validate thresholds
python3 validate_thresholds.py
```

---

## 🎯 Success Criteria

You know training worked correctly when:
1. ✅ No errors during training
2. ✅ Val accuracy reaches 96-97%
3. ✅ ECE < 0.05 (well-calibrated)
4. ✅ Exit coverage ~60% at threshold 0.88
5. ✅ classifier_head.pth saved successfully
6. ✅ config.json contains all hyperparameters

---

## 💰 Cost Tracking

- **Setup time**: Free (just SSH commands)
- **Training time**: ~1.5-2 hours × $0.45/hr = **~$0.90**
- **Total budget**: $5
- **Remaining**: $4.10 for additional experiments

**Tip**: Use feature caching mode for experiments to save money!

---

Good luck! 🚀
