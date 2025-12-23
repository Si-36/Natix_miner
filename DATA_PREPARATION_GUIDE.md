# ROADWork Data Preparation Guide

## 🎯 Goal
Aggressively expand training beyond NATIX by adding **ROADWork (ICCV 2025)** + small public roadwork datasets to maximize accuracy and robustness on work zone detection.

**Why this matters**: The ROADWork paper shows **32.5% precision improvement** and **12.8× higher discovery rate** on work zones after fine-tuning on diverse work zone data. Combining NATIX (Europe) + ROADWork (US) + extras gives maximum geographical and scenario coverage.

---

## 📊 Target Datasets

### 1. NATIX Official (Already Have)
- **Source**: Your local NATIX data
- **Location**: `data/natix_official/`
- **Size**: ~10,000 train, ~2,500 val
- **Coverage**: Europe-centric, crowdsourced

### 2. ROADWork Dataset (MUST ADD)
- **Source**: CMU ICCV 2025 dataset
- **Paper**: https://arxiv.org/abs/2406.07661
- **GitHub**: https://github.com/anuragxel/roadwork-dataset
- **Website**: https://cs.cmu.edu/~roadwork/
- **Size**: ~5,000 work zones in 18 US cities
- **Coverage**: US-centric, diverse weather/lighting/layouts

### 3. Roboflow Roadwork (OPTIONAL BUT RECOMMENDED)
- **Source**: https://universe.roboflow.com/workzone/roadwork
- **Size**: ~500-1,000 images
- **Coverage**: Additional edge cases, camera types

---

## 🚀 Step-by-Step Setup

### Step 1: Download ROADWork Dataset

```bash
cd /home/sina/projects/miner_b/streetvision_cascade

# Create download info
python3 prepare_roadwork_data.py --download_roadwork
```

**This will show**:
```
DOWNLOADING ROADWORK DATASET (ICCV 2025)
================================================================================

📦 ROADWork Dataset Info:
   Paper: https://arxiv.org/abs/2406.07661
   GitHub: https://github.com/anuragxel/roadwork-dataset
   Output: data/roadwork_iccv

⚠️  MANUAL STEP REQUIRED:
   1. Visit: https://github.com/anuragxel/roadwork-dataset
   2. Follow their download instructions (likely Google Drive or similar)
   3. Download the dataset to: data/roadwork_iccv/raw/
   4. Extract all images and annotations
   5. Re-run this script with --process_roadwork to convert to our format
```

**Manual steps**:

1. Visit https://github.com/anuragxel/roadwork-dataset

2. Look for download links (probably Google Drive, OneDrive, or direct download)

3. Download **train split** (and optionally val split)

4. Extract to `data/roadwork_iccv/raw/`:
   ```bash
   mkdir -p data/roadwork_iccv/raw
   # Extract downloaded files here
   ```

5. Expected structure after extraction:
   ```
   data/roadwork_iccv/raw/
   ├── images/
   │   ├── city1/
   │   ├── city2/
   │   └── ...
   └── annotations/
       ├── train.json  (or similar)
       └── val.json
   ```

---

### Step 2: Process ROADWork to Binary Labels

```bash
python3 prepare_roadwork_data.py --process_roadwork
```

**What this does**:
- Reads ROADWork annotations (JSON files)
- Converts to binary labels:
  - `label = 1` if work zone present (cones, barriers, arrow boards, signs, workers)
  - `label = 0` if clean road (no work zone)
- Creates `data/roadwork_iccv/train_labels.csv` in our format: `image_path,label`

**Expected output**:
```
PROCESSING ROADWORK DATASET
================================================================================
✅ Found 12 annotation files
✅ Processed ROADWork dataset:
   Train: 4523 samples -> data/roadwork_iccv/train_labels.csv
   Val:   1201 samples -> data/roadwork_iccv/val_labels.csv
   Work zones (train): 3890 (86.0%)
   Work zones (val):   1032 (85.9%)
```

---

### Step 3: Download Roboflow Extras (Optional)

```bash
python3 prepare_roadwork_data.py --download_extra
```

**Manual steps**:

1. Visit https://universe.roboflow.com/workzone/roadwork

2. Click "Download Dataset"

3. Choose format: **COCO JSON** or **Pascal VOC**

4. Download and extract to `data/roadwork_extra/raw/`:
   ```bash
   mkdir -p data/roadwork_extra/raw
   # Extract downloaded files here
   ```

5. Process:
   ```bash
   python3 prepare_roadwork_data.py --process_extra
   ```

**Expected output**:
```
PROCESSING ROBOFLOW ROADWORK DATASETS
================================================================================
✅ Found 847 images
✅ Processed Roboflow roadwork:
   Train: 847 samples -> data/roadwork_extra/train_labels.csv
   All labeled as roadwork=1 (focused dataset)
```

---

### Step 4: Create Documentation

```bash
python3 prepare_roadwork_data.py --create_docs
```

This creates `DATA_SOURCES.md` documenting all datasets.

---

### Step 5: Verify Directory Structure

```bash
tree -L 3 data/
```

**Expected structure**:
```
data/
├── natix_official/
│   ├── train/
│   ├── train_labels.csv
│   ├── val/
│   └── val_labels.csv
├── roadwork_iccv/
│   ├── raw/  (original download)
│   ├── train_images/
│   ├── train_labels.csv
│   ├── val_images/
│   └── val_labels.csv
└── roadwork_extra/
    ├── raw/  (original download)
    ├── train_images/
    └── train_labels.csv
```

---

## 🏋️ Training with Multi-Dataset

### Standard Mode (NATIX only)
```bash
cd streetvision_cascade
python3 train_stage1_head.py --mode train --epochs 10
```

Uses: **~10,000 samples** (NATIX train only)

### Aggressive Mode (Max Data)
```bash
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```

Uses: **~15,000-16,000 samples** (NATIX + ROADWork + extras)

**What happens**:
- Loads NATIX train
- Adds ROADWork train (if available)
- Adds Roboflow extras (if available)
- Computes class weights over **combined** dataset
- Validates on **NATIX val only** (primary deployment metric)

**Expected console output**:
```
[2/7] Loading dataset...
🚀 MULTI-DATASET MODE: Combining all roadwork sources
   ✅ Adding ROADWork dataset (ICCV 2025)
   ✅ Adding extra roadwork datasets (Roboflow, etc.)

📊 Multi-Dataset Stats:
   Total samples: 15370
   natix_official: 10000 samples
   roadwork_iccv: 4523 samples
   roadwork_extra: 847 samples

📊 Class distribution:
   Class 0 (no roadwork): 6530 samples (42.5%)
   Class 1 (roadwork):    8840 samples (57.5%)
   Class weights: [0.766 0.564]
```

---

## 📈 Expected Impact

From ROADWork paper and empirical results:

### Baseline (NATIX only)
- **Val Accuracy**: 96-97%
- **Recall on US work zones**: Medium (unseen geography)
- **Recall on night/rain work zones**: Medium (fewer examples)

### With ROADWork + Extras
- **Val Accuracy**: 97-98% (higher)
- **Recall on US work zones**: **+10-15%** (ROADWork covers US extensively)
- **Recall on night/rain work zones**: **+5-10%** (ROADWork has diverse conditions)
- **ECE (Calibration)**: **Lower** (more diverse positive examples)
- **Exit Accuracy @ 0.88**: **Higher, more stable** (better confidence on edge cases)

**ROADWork paper results**:
- **+32.5% precision** on work zone detection
- **12.8× higher discovery rate** in global imagery
- Significantly better generalization to unseen work zone types

---

## 🔍 Quality Control Checklist

Before training with `--use_extra_roadwork`, verify:

### 1. Data Integrity
```bash
# Check all CSVs exist
ls data/natix_official/train_labels.csv
ls data/roadwork_iccv/train_labels.csv
ls data/roadwork_extra/train_labels.csv

# Check sample counts
wc -l data/*/train_labels.csv
```

### 2. Label Distribution
```bash
# Quick stats
python3 << 'EOF'
import pandas as pd

natix = pd.read_csv("data/natix_official/train_labels.csv", header=None, names=['path', 'label'])
roadwork = pd.read_csv("data/roadwork_iccv/train_labels.csv", header=None, names=['path', 'label'])
extra = pd.read_csv("data/roadwork_extra/train_labels.csv", header=None, names=['path', 'label'])

print(f"NATIX: {len(natix)} samples, {natix['label'].mean()*100:.1f}% positive")
print(f"ROADWork: {len(roadwork)} samples, {roadwork['label'].mean()*100:.1f}% positive")
print(f"Extra: {len(extra)} samples, {extra['label'].mean()*100:.1f}% positive")

combined_pos_rate = (natix['label'].sum() + roadwork['label'].sum() + extra['label'].sum()) / (len(natix) + len(roadwork) + len(extra))
print(f"\nCombined: {len(natix)+len(roadwork)+len(extra)} samples, {combined_pos_rate*100:.1f}% positive")
EOF
```

### 3. Visual Inspection
```bash
# Sample 10 random images from each dataset and visually verify labels
```

---

## 💰 Training Time & Cost Estimates

### NATIX only (10K samples)
- **Time**: ~1.5-2 hours
- **Cost**: ~$0.90 @ $0.45/hr

### NATIX + ROADWork + Extras (15K samples)
- **Time**: ~2-2.5 hours (+50% data)
- **Cost**: ~$1.20 @ $0.45/hr

**Still well within $5 budget!**

---

## 🛟 Troubleshooting

### Problem: "⚠️ ROADWork not found"
**Solution**:
- Check that you downloaded and extracted to `data/roadwork_iccv/raw/`
- Run `--process_roadwork` to create the `train_labels.csv`

### Problem: "FileNotFoundError: image file not found"
**Solution**:
- Check that images are in `data/roadwork_iccv/train_images/` (not still in `raw/`)
- The processing script should have copied/linked them automatically

### Problem: Class imbalance warning (>90% positive)
**Solution**:
- This is expected for ROADWork (focused on work zones)
- Class weights will handle it automatically
- NATIX provides the negatives for balance

### Problem: Training is slower with more data
**Solution**:
- This is expected (+50% data = +50% time)
- Use feature caching mode for fast iteration:
  ```bash
  python3 train_stage1_head.py --mode extract_features --use_extra_roadwork
  python3 train_stage1_head.py --mode train_cached --epochs 20
  ```

---

## 📚 References

- **ROADWork Paper**: Ghosh et al. "ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones" ICCV 2025
- **ROADWork GitHub**: https://github.com/anuragxel/roadwork-dataset
- **NATIX**: https://github.com/natixnetwork/streetvision-subnet
- **Roboflow**: https://universe.roboflow.com/workzone/roadwork

---

Last updated: 2025-12-23
