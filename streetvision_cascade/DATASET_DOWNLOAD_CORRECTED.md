# Dataset Download Guide (2025 Pro - CORRECTED)

## Critical Rule: Validation Set Purity
**NEVER mix external datasets into NATIX val.** Always evaluate on NATIX val only.

---

## Dataset Roles (Corrected Understanding)

| Dataset | Role | Label Strategy |
|---------|------|----------------|
| **NATIX** | Primary train/val, deployment distribution | Balanced binary (0/1) |
| **ROADWork** | Best work-zone ground truth | Positive-rich (mostly 1) |
| **Open Images V7** | Global diversity booster | **POSITIVES ONLY** (all 1) |
| **Roboflow** | Fast curated positives | All 1 |
| **GTSRB Class 25** | EU sign booster | All 1 |
| **BDD100K (optional)** | Hard negatives, realism | Mainly 0 |

**Key Insight**: External datasets are **positives boosters**. Your negatives come from NATIX + normal driving (BDD100K).

---

## Step-by-Step Download (Laptop)

### 0. NATIX (Already Have)
```bash
# Verify structure
ls data/natix_official/train/train_labels.csv
ls data/natix_official/val/val_labels.csv

# Check no leakage between train/val
# This is your ONLY validation set - keep it pure!
```

---

### 1. ROADWork (⭐⭐⭐⭐⭐ Best External)

**Manual Download from CMU KiltHub**:

```bash
cd ~/projects/miner_b/streetvision_cascade

# Create directories
mkdir -p data/roadwork_iccv/raw

# Visit: https://github.com/anuragxel/roadwork-dataset
# Follow their README to find CMU KiltHub link
# They list exact ZIPs: images.zip, annotations.zip, sem_seg_labels.zip, etc.

# Download ZIPs manually to: data/roadwork_iccv/raw/
# Example (replace with actual KiltHub URLs):
wget -P data/roadwork_iccv/raw/ <KILTHUB_URL>/images.zip
wget -P data/roadwork_iccv/raw/ <KILTHUB_URL>/annotations.zip

# Unzip following their directory structure
cd data/roadwork_iccv/raw
unzip images.zip
unzip annotations.zip
cd ../../..

# Structure should match their repo requirements:
# data/roadwork_iccv/raw/
#   scene/
#     images/
#     annotations/
```

**Processing** (after fixing the parser):
```bash
python3 prepare_roadwork_data.py --process_roadwork
```

---

### 2. Roboflow Work Zone (⭐⭐⭐⭐ Fast Positives)

```bash
# Manual download from: https://universe.roboflow.com/workzone/roadwork
# Choose: COCO JSON format
# Download ZIP to: ~/Downloads/roboflow_roadwork.zip

mkdir -p data/roadwork_extra/raw
unzip ~/Downloads/roboflow_roadwork.zip -d data/roadwork_extra/raw/

# Process (labels all as 1)
python3 prepare_roadwork_data.py --process_extra
```

---

### 3. Open Images V7 (⭐⭐⭐⭐ POSITIVES BOOSTER)

**CORRECTED VERSION** - Only use as positives, don't create fake negatives:

```bash
pip install fiftyone

# Run corrected script (see below)
python3 download_open_images_positives_only.py
```

**Key Change**: Label ALL downloaded images as `1` (positives), because class-filtered download gives you positives.

---

### 4. GTSRB Class 25 (⭐⭐⭐⭐ EU Signs)

**CORRECTED VERSION** - Properly convert .ppm to .png:

```bash
pip install kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb_class25/raw/

# Run corrected converter (see below)
python3 convert_gtsrb_class25.py
```

---

### 5. BDD100K (⭐⭐⭐ Optional - Hard Negatives)

Only if you have bandwidth and want hard negatives:

```bash
# Register at: https://bdd-data.berkeley.edu/
# Download: 100K Images + Detection Labels
# Filter for negatives (images without work zones)
```

---

## Training Strategy (Corrected)

### Baseline (Safe)
```bash
python3 train_stage1_head.py --mode train --epochs 10
# Train: NATIX train only (~10K)
# Val: NATIX val only (~2.5K)
```

### Aggressive (Max Positives)
```bash
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
# Train: NATIX train + ROADWork + Open Images + Roboflow + GTSRB (all positives)
# Val: NATIX val only (NEVER mix external datasets here)
```

**Why This Works**:
- NATIX provides balanced binary distribution
- External datasets add diverse positive examples
- NATIX val stays pure → metrics match deployment

---

## Expected Results (Realistic, Not Overpromised)

### Baseline (NATIX only)
- Val Accuracy: 96-97%
- Cost: ~$0.90

### Aggressive (NATIX + all positives boosters)
- Val Accuracy: **97-98%** (modest improvement)
- US work zone recall: **+5-10%** (not +32.5%)
- EU sign recall: **+5-10%**
- Cost: ~$1.50

**Still under $5 budget.**

---

## What NOT to Do

❌ Don't claim Open Images gives you "balanced binary dataset"
❌ Don't put ROADWork/Roboflow/OpenImages into NATIX val
❌ Don't rename .ppm to .png without converting
❌ Don't expect +32.5% precision boost (that's from ROADWork's discovery pipeline)

---

Last updated: 2025-12-23 (Corrected)
