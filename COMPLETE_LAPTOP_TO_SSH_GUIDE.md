# Complete Laptop to SSH Training Guide (2025 Pro)

**Total Time**: ~3-4 hours (1-2 hrs download, 1-2 hrs transfer, training on SSH)
**Total Cost**: ~$2.40 (baseline $0.90 + aggressive $1.50)

---

## 📋 Phase 1: Laptop Preparation (1-2 hours)

### Step 1: Verify NATIX Dataset ✅

```bash
cd ~/projects/miner_b/streetvision_cascade

# Verify NATIX is correct
python3 verify_datasets.py --check_natix
```

**Expected Output**:
```
✅ NATIX dataset OK!
   Train: ~10000 samples
   Val:   ~2500 samples
```

**If errors**: Fix NATIX structure before proceeding.

---

### Step 2: Download ROADWork (30 min) 🔥 BEST DATASET

**Manual Download**:
1. Visit: https://github.com/anuragxel/roadwork-dataset
2. Find CMU KiltHub link in their README
3. Download these ZIPs:
   - `images.zip` (largest file)
   - `annotations.zip`
   - (Optional) `sem_seg_labels.zip`

**Save to**: `~/Downloads/`

**Process**:
```bash
cd ~/projects/miner_b/streetvision_cascade

# Create directories
mkdir -p data/roadwork_iccv/raw

# Move ZIPs
mv ~/Downloads/images.zip data/roadwork_iccv/raw/
mv ~/Downloads/annotations.zip data/roadwork_iccv/raw/

# Unzip (follow their directory structure)
cd data/roadwork_iccv/raw
unzip images.zip
unzip annotations.zip
cd ../../..

# Process to binary labels
python3 prepare_roadwork_data.py --process_roadwork
```

**Expected Output**:
```
✅ TRAIN: ~4000-5000 samples
   Work zones: ~3500 (70-80%)
   Clean roads: ~500-1000 (20-30%)
```

**Checkpoint**:
```bash
ls data/roadwork_iccv/train_labels.csv  # Should exist
head -5 data/roadwork_iccv/train_labels.csv  # Check format
```

---

### Step 3: Download Open Images V7 (45 min) 🌍 GLOBAL DIVERSITY

**Install FiftyOne**:
```bash
pip install fiftyone
```

**Download (Positives Only)**:
```bash
cd ~/projects/miner_b/streetvision_cascade

# Run corrected script (treats all as positives)
python3 download_open_images_positives_only.py
```

**What happens**:
- Downloads ~2000 images with cones/barriers/signs
- All labeled as `1` (positives)
- Saves to `data/open_images/coco/data/`
- Creates `data/open_images/train_labels.csv`

**Expected Output**:
```
✅ Downloaded ~2000 images
   ALL labeled as 1 (positives)
⚠️  Remember: Your negatives come from NATIX!
```

**Checkpoint**:
```bash
ls data/open_images/train_labels.csv  # Should exist
ls data/open_images/coco/data/ | head -5  # Check images exist
```

---

### Step 4: Download Roboflow (10 min) ⚡ FAST POSITIVES

**Manual Download**:
1. Visit: https://universe.roboflow.com/workzone/roadwork
2. Click "Download Dataset"
3. Choose format: **COCO JSON**
4. Download ZIP to `~/Downloads/roboflow_roadwork.zip`

**Process**:
```bash
cd ~/projects/miner_b/streetvision_cascade

# Create directories
mkdir -p data/roadwork_extra/raw

# Unzip
unzip ~/Downloads/roboflow_roadwork.zip -d data/roadwork_extra/raw/

# Process to binary labels
python3 prepare_roadwork_data.py --process_extra
```

**Expected Output**:
```
✅ ~500-1000 samples
   ALL labeled as 1 (focused dataset)
```

**Checkpoint**:
```bash
ls data/roadwork_extra/train_labels.csv  # Should exist
wc -l data/roadwork_extra/train_labels.csv  # Check count
```

---

### Step 5: Download GTSRB Class 25 (15 min) 🇪🇺 EU SIGNS

**Install Kaggle CLI**:
```bash
pip install kaggle pillow
```

**Setup Kaggle API** (if first time):
1. Go to: https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

**Download**:
```bash
cd ~/projects/miner_b/streetvision_cascade

# Download from Kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# Unzip
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb_class25/raw/

# Convert .ppm to .png (Class 25 only)
python3 convert_gtsrb_class25.py
```

**Expected Output**:
```
✅ Converted ~600 images to PNG
   ALL labeled as 1 (EU roadwork signs)
```

**Checkpoint**:
```bash
ls data/gtsrb_class25/train_labels.csv  # Should exist
ls data/gtsrb_class25/train_images/*.png | wc -l  # Should be ~600
```

---

### Step 6: Verify All Datasets ✅ CRITICAL

```bash
cd ~/projects/miner_b/streetvision_cascade

# Verify everything is correct
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

**If ANY errors**: Fix them before proceeding to Step 7.

**Common Issues**:
- **"Image not found"**: Check CSV paths match actual files
- **"All positives"**: Normal for Open Images/Roboflow/GTSRB
- **"Too few samples"**: Re-download or check extraction

---

### Step 7: Compress Models + Data (15-30 min) 📦

```bash
cd ~/projects/miner_b

# Compress models (60GB) + data (~10-20GB)
tar -czf miner_data.tar.gz models/ streetvision_cascade/data/

# Check size
ls -lh miner_data.tar.gz
```

**Expected Size**: ~70-80 GB compressed

**What's included**:
- `models/` (60GB) - DINOv3 models you already downloaded
- `streetvision_cascade/data/natix_official/` - NATIX train/val
- `streetvision_cascade/data/roadwork_iccv/` - ROADWork
- `streetvision_cascade/data/open_images/` - Open Images V7
- `streetvision_cascade/data/roadwork_extra/` - Roboflow
- `streetvision_cascade/data/gtsrb_class25/` - GTSRB

---

## 📤 Phase 2: Transfer to SSH Server (1-2 hours)

### Step 8: Upload to SSH Server

**Option A: SCP (simple)**:
```bash
scp miner_data.tar.gz user@vast.ai:/workspace/
```

**Option B: rsync (can resume if interrupted)**:
```bash
rsync -avz --progress miner_data.tar.gz user@vast.ai:/workspace/
```

**Time estimate**:
- 10 Mbps upload: ~18 hours (do overnight)
- 50 Mbps upload: ~3.5 hours
- 100 Mbps upload: ~1.8 hours
- 1000 Mbps upload: ~11 minutes

**Tip**: Run in `screen` or `tmux` so it continues if connection drops:
```bash
screen -S upload
scp miner_data.tar.gz user@vast.ai:/workspace/
# Press Ctrl+A then D to detach
# Later: screen -r upload to reattach
```

---

## 🚀 Phase 3: SSH Server Setup (30 min)

### Step 9: Extract and Setup

**Connect to SSH**:
```bash
ssh user@vast.ai
```

**Extract data**:
```bash
cd /workspace

# Extract (takes 5-10 min for 70GB)
tar -xzf miner_data.tar.gz

# Check extraction
ls -lh models/
ls -lh streetvision_cascade/data/
```

**Clone your repo**:
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/miner_b.git
cd miner_b
```

**Move models and data into repo**:
```bash
# Move models
mv /workspace/models .

# Move data
mv /workspace/streetvision_cascade/data streetvision_cascade/

# Verify structure
ls -lh models/
ls -lh streetvision_cascade/data/
```

**Expected structure**:
```
/workspace/miner_b/
├── models/
│   └── stage1_dinov3/
│       └── dinov3-vith16plus-pretrain-lvd1689m/
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

### Step 10: Install Dependencies

```bash
cd /workspace/miner_b

# Install PyTorch + dependencies
pip install torch torchvision transformers timm datasets pillow tqdm scikit-learn pandas
```

**Verify GPU**:
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Expected**:
```
True
NVIDIA RTX A6000
```

---

### Step 11: Verify Datasets on Server

```bash
cd /workspace/miner_b/streetvision_cascade

# Verify all datasets work on SSH server
python3 verify_datasets.py --check_all
```

**Expected**: Same output as laptop verification (all ✅)

**If errors**: Check paths and file permissions.

---

## 🎓 Phase 4: Training (3-4 hours)

### Step 12: Baseline Training (NATIX Only)

```bash
cd /workspace/miner_b/streetvision_cascade

# Run baseline (establish metrics)
python3 train_stage1_head.py --mode train --epochs 10
```

**What to expect**:
```
================================================================================
FULL TRAINING MODE (with data augmentation)
================================================================================
Using device: cuda
GPU: NVIDIA RTX A6000
VRAM: 48.0 GB

[2/7] Loading dataset...
📦 NATIX-only mode (use --use_extra_roadwork for more data)
✅ timm-style augmentation enabled for training
✅ Validation: NATIX val only (primary deployment metric)

[3/7] Setting up loss function...
📊 Class distribution:
   Class 0 (no roadwork): 7500 samples (75.0%)
   Class 1 (roadwork):    2500 samples (25.0%)
   Class weights: [0.67 2.00]

[4/7] Setting up optimizer...
✅ AdamW optimizer (lr_head=1e-4, lr_backbone=1e-5)
✅ Cosine LR schedule with 1.0 epoch warmup
✅ Gradient clipping: max_norm=1.0

[5/7] Training...
Epoch 1/10: Train Acc: 92.34%, Val Acc: 94.56%
  ECE: 0.0421, Exit@0.88: 58.3% @ 96.2% acc
  ✅ Saved best checkpoint

Epoch 2/10: Train Acc: 94.12%, Val Acc: 95.87%
...

[7/7] Training complete! Best Val Acc: 96.78%
✅ Model saved to: outputs/stage1_head/classifier_head.pth
```

**Metrics to record**:
- Best Val Accuracy: ~96-97%
- ECE: ~0.03-0.05
- Exit Coverage @ 0.88: ~55-65%
- Exit Accuracy @ 0.88: ~96-98%

**Time**: ~1.5-2 hours
**Cost**: ~$0.90 @ $0.45/hr

---

### Step 13: Aggressive Training (Multi-Dataset)

```bash
cd /workspace/miner_b/streetvision_cascade

# Run aggressive mode (max data)
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```

**What to expect**:
```
[2/7] Loading dataset...
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

📊 Class distribution:
   Class 0 (no roadwork): 8500 samples (47.3%)
   Class 1 (roadwork):    9470 samples (52.7%)
   Class weights: [0.588 0.527]

Epoch 1/15: Train Acc: 93.45%, Val Acc: 95.23%
...
[7/7] Training complete! Best Val Acc: 97.89%
```

**Metrics to record**:
- Best Val Accuracy: ~97-98% (+1-2% vs baseline)
- ECE: ~0.02-0.04 (better calibration)
- Exit Coverage @ 0.88: ~58-68%
- Exit Accuracy @ 0.88: ~97-99%

**Time**: ~2.5-3 hours
**Cost**: ~$1.50 @ $0.45/hr

---

## 📊 Phase 5: Compare Results

### Baseline vs Aggressive Comparison

| Metric | Baseline (NATIX only) | Aggressive (Multi-dataset) | Improvement |
|--------|----------------------|---------------------------|-------------|
| **Train Samples** | ~10,000 | ~18,000 | +80% data |
| **Val Accuracy** | 96-97% | 97-98% | +1-2% |
| **ECE** | 0.03-0.05 | 0.02-0.04 | Better calibration |
| **Exit Coverage @ 0.88** | 55-65% | 58-68% | More confident |
| **Exit Accuracy @ 0.88** | 96-98% | 97-99% | Higher quality exits |
| **Training Time** | ~2 hrs | ~3 hrs | +50% time |
| **Cost** | ~$0.90 | ~$1.50 | +$0.60 |

**Expected Improvements**:
- ✅ +1-2% validation accuracy
- ✅ Better calibration (lower ECE)
- ✅ +5-10% recall on US work zones (ROADWork)
- ✅ +5-10% recall on EU signs (GTSRB)
- ✅ More confident predictions on edge cases
- ✅ Better generalization to unseen scenarios

**Total Cost**: ~$2.40 (well under $5 budget!)

---

## ✅ Final Checklist

### On Laptop (Before Transfer):
- [ ] NATIX verified (train + val)
- [ ] ROADWork downloaded and processed (~4K samples)
- [ ] Open Images V7 downloaded (~2K samples)
- [ ] Roboflow downloaded and processed (~800 samples)
- [ ] GTSRB Class 25 downloaded and converted (~600 samples)
- [ ] `verify_datasets.py --check_all` passed
- [ ] `miner_data.tar.gz` created (~70-80 GB)

### On SSH Server (Before Training):
- [ ] Data transferred and extracted
- [ ] Repo cloned
- [ ] Models and data moved to correct locations
- [ ] Dependencies installed
- [ ] GPU verified (NVIDIA RTX A6000)
- [ ] `verify_datasets.py --check_all` passed on server

### After Training:
- [ ] Baseline run completed (~96-97% accuracy)
- [ ] Aggressive run completed (~97-98% accuracy)
- [ ] Results compared (expect +1-2% improvement)
- [ ] Best model saved: `outputs/stage1_head/classifier_head.pth`
- [ ] Config saved: `outputs/stage1_head/config.json`

---

## 🔥 Quick Reference Commands

**Laptop**:
```bash
# Verify NATIX
cd ~/projects/miner_b/streetvision_cascade && python3 verify_datasets.py --check_natix

# Verify all datasets
python3 verify_datasets.py --check_all

# Compress
cd ~/projects/miner_b && tar -czf miner_data.tar.gz models/ streetvision_cascade/data/

# Transfer
scp miner_data.tar.gz user@vast.ai:/workspace/
```

**SSH Server**:
```bash
# Extract
cd /workspace && tar -xzf miner_data.tar.gz

# Setup
git clone https://github.com/YOUR_USERNAME/miner_b.git
cd miner_b && mv /workspace/models /workspace/streetvision_cascade/data .

# Install
pip install torch torchvision transformers timm datasets pillow tqdm scikit-learn pandas

# Verify
cd streetvision_cascade && python3 verify_datasets.py --check_all

# Train baseline
python3 train_stage1_head.py --mode train --epochs 10

# Train aggressive
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```

---

## 🛟 Troubleshooting

### "FileNotFoundError: data/roadwork_iccv/train_images/..."
**Fix**: Check CSV paths match actual files. Run `verify_datasets.py --check_extra`

### "Training shows 10000 samples with --use_extra_roadwork"
**Fix**: External CSVs not found. Check paths exist:
```bash
ls data/roadwork_iccv/train_labels.csv
ls data/open_images/train_labels.csv
ls data/roadwork_extra/train_labels.csv
ls data/gtsrb_class25/train_labels.csv
```

### "SCP connection dropped during transfer"
**Fix**: Use rsync instead (can resume):
```bash
rsync -avz --partial --progress miner_data.tar.gz user@vast.ai:/workspace/
```

### "Out of GPU memory"
**Fix**: Reduce batch size in config or use gradient accumulation (already enabled)

---

**Total Timeline**: ~6-8 hours (1-2 hrs download, 1-2 hrs transfer, 3-4 hrs training)
**Total Cost**: ~$2.40 (well under $5 budget!)

**You're ready to go! Follow the steps in order and you'll have a production-grade model.** 🚀

Last updated: 2025-12-23
