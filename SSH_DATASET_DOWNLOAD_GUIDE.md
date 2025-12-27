# 🚀 Complete SSH Dataset Download Guide (2025 Production)

## 📋 Overview

This guide downloads **all datasets directly on SSH** (no transfer needed). Follow this step-by-step to build the complete training pipeline.

---

## 🎯 Why Two Roadwork Folders?

You have **TWO** roadwork datasets for different purposes:

| Folder | Source | Size | Purpose |
|--------|--------|------|---------|
| **`roadwork_iccv/`** | CMU ROADWork ICCV 2025 | ~5,000 samples | **Best quality** - US work zones, diverse conditions |
| **`roadwork_extra/`** | Roboflow Universe | ~500 samples | **Quick extras** - Additional edge cases |

**Why both?**
- ROADWork ICCV is state-of-the-art (32.5% precision boost in paper)
- Roboflow adds quick curated positives
- Together: Maximum coverage (US + Europe + edge cases)

---

## 📦 Complete Dataset List

### ✅ Already Downloaded (Check First)
1. **NATIX Official** - `data/natix_official/` (~10K train, ~2.5K val)
2. **ROADWork ICCV** - `data/roadwork_iccv/` (~5K samples)
3. **Roboflow Extra** - `data/roadwork_extra/` (~500 samples)
4. **Open Images V7** - `data/open_images/` (~2K positives)
5. **GTSRB Class 25** - `data/gtsrb_class25/` (~1.5K EU signs)
6. **Kaggle Road Issues** - `data/kaggle_road_issues/` (~6K negatives)

### 🔄 Need to Download/Verify on SSH
- NATIX (from Hugging Face)
- ROADWork (if not already processed)
- Open Images (if not complete)
- GTSRB (if not complete)

---

## 🚀 Step-by-Step SSH Setup

### **Step 0: SSH Connection & Environment**

```bash
# SSH into your server
ssh ubuntu@your-server-ip

# Navigate to project
cd ~/Natix_miner/streetvision_cascade

# Activate virtual environment
source .venv/bin/activate

# Verify Python and packages
python3 --version  # Should be 3.10+
pip list | grep torch  # Should see torch, transformers, etc.
```

---

### **Step 1: Download NATIX Official Dataset** ⭐ PRIMARY

**Source**: Hugging Face `natix-network-org/roadwork`

```bash
cd ~/Natix_miner/streetvision_cascade

# Create directory
mkdir -p data/natix_official/data

# Download using Hugging Face datasets library
python3 << 'PY'
from datasets import load_dataset
from pathlib import Path

print("📥 Downloading NATIX official dataset from Hugging Face...")
print("   This may take 10-20 minutes (~8GB)")

# Load dataset
dataset = load_dataset("natix-network-org/roadwork", split="train")

# Save as parquet (original format)
output_dir = Path("data/natix_official/data")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert to parquet files (batched)
dataset.to_parquet(str(output_dir / "train.parquet"), batch_size=1000)

print(f"✅ NATIX dataset downloaded to: {output_dir}")
print(f"   Samples: {len(dataset)}")
PY

# Convert parquet to images + CSV (if not already done)
python3 convert_natix_parquet.py

# Verify
ls -lh data/natix_official/train/ | head -5
wc -l data/natix_official/train_labels.csv
```

**Expected Output**:
```
✅ NATIX dataset downloaded
   Train: ~6,000-10,000 images
   Val: ~2,000-2,500 images
   Positive rate: ~20-25%
```

---

### **Step 2: Download ROADWork ICCV 2025** ⭐ BEST EXTERNAL

**Source**: CMU ROADWork Dataset (ICCV 2025)
- **Paper**: https://arxiv.org/abs/2406.07661
- **GitHub**: https://github.com/anuragxel/roadwork-dataset
- **Website**: https://cs.cmu.edu/~roadwork/

#### **Option A: Direct Download (If Available)**

```bash
cd ~/Natix_miner/streetvision_cascade

# Create directory
mkdir -p data/roadwork_iccv/raw

# Check GitHub for download links
# Visit: https://github.com/anuragxel/roadwork-dataset
# Look for: CMU KiltHub, Google Drive, or direct download links

# Example (replace with actual URLs from GitHub):
wget -P data/roadwork_iccv/raw/ <KILTHUB_URL>/images.zip
wget -P data/roadwork_iccv/raw/ <KILTHUB_URL>/annotations.zip
wget -P data/roadwork_iccv/raw/ <KILTHUB_URL>/sem_seg_labels.zip

# Extract
cd data/roadwork_iccv/raw
unzip images.zip
unzip annotations.zip
cd ../../..
```

#### **Option B: Clone GitHub Repo (If Dataset Included)**

```bash
cd ~/Natix_miner/streetvision_cascade

# Clone repo (may include dataset or download scripts)
git clone https://github.com/anuragxel/roadwork-dataset.git /tmp/roadwork-repo

# Check README for download instructions
cat /tmp/roadwork-repo/README.md

# Follow their instructions (usually involves:
# 1. Requesting access to CMU KiltHub
# 2. Downloading ZIPs manually
# 3. Extracting to specific structure)
```

#### **Process ROADWork to Binary Labels**

```bash
# After downloading raw data
python3 prepare_roadwork_data.py --process_roadwork

# Verify
ls -lh data/roadwork_iccv/train_labels.csv
head -5 data/roadwork_iccv/train_labels.csv
```

**Expected Output**:
```
✅ Processed ROADWork dataset:
   Train: ~4,000-5,000 samples
   Val: ~1,000-2,000 samples
   Work zones: ~80-90% positive
```

---

### **Step 3: Download Roboflow Roadwork Extras** (Optional but Recommended)

**Source**: https://universe.roboflow.com/workzone/roadwork

```bash
cd ~/Natix_miner/streetvision_cascade

# Create directory
mkdir -p data/roadwork_extra/raw

# Manual download required:
# 1. Visit: https://universe.roboflow.com/workzone/roadwork
# 2. Click "Download Dataset"
# 3. Choose format: COCO JSON or Pascal VOC
# 4. Download ZIP

# Upload ZIP to SSH (or download directly on SSH if possible)
# Then extract:
unzip ~/Downloads/roboflow_roadwork.zip -d data/roadwork_extra/raw/

# Process
python3 prepare_roadwork_data.py --process_extra

# Verify
ls -lh data/roadwork_extra/train_labels.csv
wc -l data/roadwork_extra/train_labels.csv
```

**Expected Output**:
```
✅ Processed Roboflow roadwork:
   Train: ~500-1,000 samples
   All labeled as roadwork=1 (focused dataset)
```

---

### **Step 4: Download Open Images V7 (Positives Only)**

```bash
cd ~/Natix_miner/streetvision_cascade

# Download script (no MongoDB needed)
python3 download_open_images_positives_only_no_mongo.py

# This downloads ~2,000 roadwork-related images
# All labeled as positive (label=1)

# Verify
ls -lh data/open_images/images/ | head -5
wc -l data/open_images/train_labels.csv
```

**Expected Output**:
```
✅ Open Images positives downloaded:
   Train: ~2,000 samples
   All labeled as roadwork=1
```

---

### **Step 5: Download GTSRB Class 25 (EU Roadwork Signs)**

```bash
cd ~/Natix_miner/streetvision_cascade

# Download GTSRB dataset
mkdir -p data/gtsrb_class25/raw
cd data/gtsrb_class25/raw

# Download from official source
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

# Extract
unzip GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Test_Images.zip

cd ../../..

# Convert Class 25 to binary labels
python3 convert_gtsrb_class25.py

# Verify
ls -lh data/gtsrb_class25/train_images/ | head -5
wc -l data/gtsrb_class25/train_labels.csv
```

**Expected Output**:
```
✅ GTSRB Class 25 converted:
   Train: ~1,500 samples
   All labeled as roadwork=1 (EU roadwork signs)
```

---

### **Step 6: Download Kaggle Road Issues (Negatives)**

```bash
cd ~/Natix_miner/streetvision_cascade

# Download Kaggle dataset (requires Kaggle API)
# Setup Kaggle API first:
pip install kaggle
# Then configure: https://www.kaggle.com/docs/api

# Download dataset
kaggle datasets download -d <kaggle-dataset-id> -p data/kaggle_road_issues/raw

# Or manual download:
# 1. Visit: https://www.kaggle.com/datasets/your-dataset
# 2. Download ZIP
# 3. Extract to data/kaggle_road_issues/raw/

# Convert to binary labels (all negatives)
python3 convert_kaggle_road_issues.py

# Verify
ls -lh data/kaggle_road_issues/images/ | head -5
wc -l data/kaggle_road_issues/train_labels.csv
```

**Expected Output**:
```
✅ Kaggle Road Issues converted:
   Train: ~6,000 samples
   All labeled as 0 (road problems ≠ roadwork)
```

---

## ✅ **Step 7: Verify All Datasets**

```bash
cd ~/Natix_miner/streetvision_cascade

# Run comprehensive verification
python3 verify_datasets.py --check_all

# Expected output:
# ✅ NATIX Official: 6,251 train, 2,298 val
# ✅ ROADWork ICCV: 4,523 train, 1,201 val
# ✅ Roboflow Extra: 847 train
# ✅ Open Images: 2,000 train
# ✅ GTSRB Class 25: 1,500 train
# ✅ Kaggle Road Issues: 6,113 train
```

---

## 📊 **Final Dataset Summary**

After all downloads complete:

| Dataset | Train | Val | Label Strategy | Purpose |
|---------|-------|-----|----------------|---------|
| **NATIX** | ~6,000-10,000 | ~2,000-2,500 | Balanced (20-25% pos) | **Primary** - Deployment distribution |
| **ROADWork ICCV** | ~4,500-5,000 | ~1,000-2,000 | Positive-rich (80-90% pos) | **Best external** - US work zones |
| **Roboflow Extra** | ~500-1,000 | - | All positive (100%) | Quick curated positives |
| **Open Images** | ~2,000 | - | All positive (100%) | Global diversity |
| **GTSRB Class 25** | ~1,500 | - | All positive (100%) | EU roadwork signs |
| **Kaggle Road Issues** | ~6,000 | - | All negative (0%) | Hard negatives |

**Total Training Samples**: ~19,000-25,000

---

## 🏋️ **Training Modes**

### **Standard Mode (NATIX Only)**
```bash
python3 train_stage1_head.py --mode train --epochs 10
```
- Uses: NATIX train (~6K-10K samples)
- Validates on: NATIX val only
- Best for: Baseline, quick iteration

### **Aggressive Mode (All Datasets)**
```bash
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```
- Uses: NATIX + ROADWork + Roboflow + Open Images + GTSRB (~19K-25K samples)
- Validates on: NATIX val only (keeps deployment metric)
- Best for: Maximum accuracy, production deployment

---

## 🔍 **ROADWork Deep Research**

### **What ROADWork Provides**

1. **Geographic Coverage**: 18 US cities (complements NATIX Europe)
2. **Diverse Conditions**: Day, night, rain, snow, unusual layouts
3. **Rich Annotations**: Cones, barriers, arrow boards, signs, workers, lane shifts
4. **Temporal Context**: Video sequences (not just single frames)

### **Why ROADWork Matters**

- **Paper Results**: 32.5% precision improvement, 12.8× higher discovery rate
- **Benchmark Quality**: ICCV 2025 accepted dataset
- **Complementary**: US work zones + European NATIX = global coverage

### **ROADWork Structure (After Processing)**

```
data/roadwork_iccv/
├── raw/                    # Original download
│   ├── images/            # ~8,500 images
│   ├── annotations/       # COCO JSON files
│   └── sem_seg_labels/    # Semantic segmentation masks
├── train_images/          # Processed train images (if moved)
├── train_labels.csv       # Binary labels: image_path,label
└── val_labels.csv         # Binary labels: image_path,label
```

### **ROADWork Label Mapping**

```python
# Binary conversion logic:
label = 1 if (
    work_zone_present == True OR
    cones > 0 OR
    barriers > 0 OR
    arrow_boards > 0 OR
    work_zone_signs > 0 OR
    workers > 0
) else 0
```

---

## 🐛 **Troubleshooting**

### **ROADWork Download Fails**

**Problem**: Can't find download links on GitHub

**Solution**:
1. Check GitHub README: https://github.com/anuragxel/roadwork-dataset
2. Look for CMU KiltHub link (may require academic email)
3. Contact authors if link expired
4. Alternative: Use Roboflow + Open Images (still good coverage)

### **ROADWork Processing Fails**

**Problem**: `FileNotFoundError: annotations/instances_train.json`

**Solution**:
```bash
# Check actual structure
ls -R data/roadwork_iccv/raw/

# Adjust parser in prepare_roadwork_data.py if format differs
# Common formats: COCO JSON, Pascal VOC XML, YOLO TXT
```

### **NATIX Download Slow**

**Problem**: Hugging Face download stalls

**Solution**:
```bash
# Use hf_transfer (faster)
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Retry download
python3 convert_natix_parquet.py
```

---

## 📝 **Next Steps After Download**

1. **Verify All Datasets**: `python3 verify_datasets.py --check_all`
2. **Train Baseline**: `python3 train_stage1_head.py --mode train --epochs 10`
3. **Validate Thresholds**: `python3 validate_thresholds.py`
4. **Test Cascade**: `python3 test_cascade_small.py`
5. **Train Aggressive**: `python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork`

---

## 🎯 **Key Takeaways**

1. **Two Roadwork Folders Are Correct**:
   - `roadwork_iccv/` = Best quality (CMU ICCV 2025)
   - `roadwork_extra/` = Quick extras (Roboflow)

2. **Download on SSH** = Faster than transfer (especially for 34GB)

3. **ROADWork is Critical** = 32.5% precision boost in paper

4. **Always Validate on NATIX Val** = Keeps deployment metric pure

5. **Combined Dataset** = ~19K-25K samples = Maximum accuracy

---

**Last Updated**: 2025-12-24
**Status**: Production-ready for SSH deployment

