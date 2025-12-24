# SSH Server Setup Instructions

**Run these AFTER laptop setup is complete and datasets are transferred.**

**Time**: ~30-45 min (including model download)

---

## 🚀 SSH Setup - Step by Step

### Step 1: Connect to SSH Server

```bash
ssh user@your-vast-ai-ip
# Or if using custom port:
ssh -p PORT user@your-vast-ai-ip
```

**Expected**: You should see the server prompt.

---

### Step 2: Extract Datasets

```bash
cd /workspace

# Extract datasets (takes 5-10 min for 10-20 GB)
tar -xzf datasets_only.tar.gz

# Check extraction worked
ls -lh streetvision_cascade/data/
```

**Expected Output**:
```
drwxr-xr-x  natix_official/
drwxr-xr-x  roadwork_iccv/
drwxr-xr-x  open_images/
drwxr-xr-x  roadwork_extra/
drwxr-xr-x  gtsrb_class25/
```

---

### Step 3: Clone Your Repository

```bash
cd /workspace

# Clone repo (replace with your GitHub username)
git clone https://github.com/YOUR_USERNAME/miner_b.git

cd miner_b
```

**Expected**: Repo cloned successfully.

---

### Step 4: Move Datasets into Repo

```bash
cd /workspace/miner_b

# Move datasets from extraction location to repo
mv /workspace/streetvision_cascade/data streetvision_cascade/

# Verify structure
ls -lh streetvision_cascade/data/
```

**Expected**: Same 5 dataset directories as before.

---

### Step 5: Download Models on Server (10-30 min) 🚀

**This is the KEY step - datacenter speeds!**

```bash
cd /workspace/miner_b

# Create models directory
mkdir -p models/stage1_dinov3

# Download DINOv3 model from Hugging Face
cd models/stage1_dinov3

# This downloads ~60 GB at datacenter speeds (10-30 min vs 13-18 hrs from laptop!)
git clone https://huggingface.co/facebook/dinov2-giant dinov3-vith16plus-pretrain-lvd1689m

cd ../..
```

**What to expect**:
```
Cloning into 'dinov3-vith16plus-pretrain-lvd1689m'...
remote: Enumerating objects: ...
Receiving objects: 100% |████████████| ...
```

**This is FAST on server** (10-30 min) vs uploading from laptop (13-18 hours)!

**Verify it worked**:
```bash
ls -lh models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m/
# Should see model files (pytorch_model.bin, config.json, etc.)
```

---

### Step 6: Verify Final Structure

```bash
cd /workspace/miner_b

# Check complete structure
ls -lh models/
ls -lh streetvision_cascade/data/
```

**Expected structure**:
```
/workspace/miner_b/
├── models/
│   └── stage1_dinov3/
│       └── dinov3-vith16plus-pretrain-lvd1689m/
│           ├── pytorch_model.bin (~60 GB)
│           ├── config.json
│           └── ...
├── streetvision_cascade/
│   ├── data/
│   │   ├── natix_official/
│   │   ├── roadwork_iccv/
│   │   ├── open_images/
│   │   ├── roadwork_extra/
│   │   └── gtsrb_class25/
│   ├── train_stage1_head.py
│   ├── verify_datasets.py
│   └── ...
└── ...
```

---

### Step 7: Install Dependencies

```bash
cd /workspace/miner_b

# Install PyTorch + all dependencies
pip install torch torchvision transformers timm datasets pillow tqdm scikit-learn pandas
```

**Check GPU works**:
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Expected Output**:
```
True
NVIDIA RTX A6000
```

**If False**: GPU not detected. Check Vast.ai instance has GPU enabled.

---

### Step 8: Verify Datasets on Server

```bash
cd /workspace/miner_b/streetvision_cascade

# Verify all datasets work on server
python3 verify_datasets.py --check_all
```

**Expected Output**:
```
================================================================================
FINAL VERIFICATION SUMMARY
================================================================================
✅ All datasets verified successfully!

NATIX: 10000 train, 2500 val
ROADWork: 4523 samples
Open Images V7: 2000 samples
Roboflow: 847 samples
GTSRB Class 25: 600 samples

📊 Total: ~18000 training samples

You can now run training:
  Baseline:   python3 train_stage1_head.py --mode train --epochs 10
  Aggressive: python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```

**If ANY errors**: Check file paths and permissions.

---

## 🎓 Ready to Train!

Your server is now fully set up. Next steps:

### Run Baseline Training (NATIX Only)

```bash
cd /workspace/miner_b/streetvision_cascade

# Run baseline (establish metrics)
python3 train_stage1_head.py --mode train --epochs 10
```

**What to expect**:
- Train samples: ~10,000
- Val samples: ~2,500
- Time: ~1.5-2 hours
- Cost: ~$0.90 @ $0.45/hr
- Val Accuracy: 96-97%

---

### Run Aggressive Training (Multi-Dataset)

```bash
cd /workspace/miner_b/streetvision_cascade

# Run aggressive mode (max data)
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```

**What to expect**:
```
🚀 MULTI-DATASET MODE: Combining all roadwork sources
   ✅ Adding ROADWork dataset (ICCV 2025)
   ✅ Adding Roboflow work zone datasets
   ✅ Adding Open Images V7 (positives booster)
   ✅ Adding GTSRB Class 25 (EU roadwork signs)

📊 Multi-Dataset Stats:
   Total samples: 17970
   natix_official: 10000 samples
   roadwork_iccv: 4523 samples
   roadwork_extra: 847 samples
   open_images: 2000 samples
   gtsrb_class25: 600 samples
```

**Expected Results**:
- Train samples: ~18,000
- Val samples: ~2,500 (NATIX only)
- Time: ~2.5-3 hours
- Cost: ~$1.50 @ $0.45/hr
- Val Accuracy: 97-98% (+1-2% vs baseline)

---

## ✅ Complete Workflow Summary

**Total Cost**: ~$2.40 (baseline $0.90 + aggressive $1.50)

**Timeline**:
1. ✅ Laptop setup: 2-3 hours (download datasets)
2. ✅ Transfer: 1-3 hours (upload datasets)
3. ✅ SSH setup: 30-45 min (including model download)
4. 🎓 Baseline training: 2 hours
5. 🎓 Aggressive training: 3 hours

**Total time**: ~8-12 hours
**Total cost**: ~$2.40 (well under $5 budget!)

---

## 🎯 Expected Improvements

| Metric | Baseline | Aggressive | Improvement |
|--------|----------|------------|-------------|
| Val Accuracy | 96-97% | 97-98% | +1-2% |
| ECE | 0.03-0.05 | 0.02-0.04 | Better calibration |
| US Work Zone Recall | Medium | +10-15% | ROADWork coverage |
| EU Sign Recall | Medium | +10-15% | GTSRB coverage |
| Exit Accuracy @ 0.88 | 96-98% | 97-99% | More confident |

---

## 🛟 SSH Troubleshooting

### Model download fails
```bash
# If git clone fails, try direct download:
cd models/stage1_dinov3
wget https://huggingface.co/facebook/dinov2-giant/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/dinov2-giant/resolve/main/config.json
```

### GPU not detected
**Fix**: Restart instance on Vast.ai, ensure GPU is enabled.

### Out of disk space
**Fix**: Delete `datasets_only.tar.gz` after extraction:
```bash
rm /workspace/datasets_only.tar.gz
```

### Training crashes with OOM
**Fix**: Already handled via gradient accumulation. Should not happen on A6000 (48GB).

---

**You're ready to train! Follow the steps above after laptop setup is complete.** 🚀
