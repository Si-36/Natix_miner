# Corrected Dataset Strategy (2025 Pro - Final Version)

## 🎯 What Changed (Corrections from Other Agent)

### ❌ My Previous Mistakes
1. **ROADWork "+32.5% precision"** - I overpromised. This is from their discovery pipeline, NOT a guaranteed boost to your binary classifier.
2. **Open Images "balanced binary dataset"** - WRONG. Class-filtered downloads give you POSITIVES, not balanced negatives.
3. **Magic download scripts** - ROADWork requires manual ZIP download from CMU KiltHub. No shortcut.
4. **GTSRB .ppm rename** - Renaming doesn't convert. Must use PIL/OpenCV.
5. **Placeholder parser** - My ROADWork parser wouldn't work with real COCO annotations.

### ✅ Corrected Understanding
- **External datasets = POSITIVES BOOSTERS** (not balanced binary)
- **Negatives come from NATIX** (your primary deployment distribution)
- **Validation = NATIX val ONLY** (never mix external datasets)
- **Manual downloads required** (no magic scripts)

---

## 📊 Dataset Roles (Corrected)

| Dataset | What It Really Is | How to Label | Why It Helps |
|---------|-------------------|--------------|--------------|
| **NATIX** | Your deployment distribution | Balanced binary (0/1) | Matches validator, provides negatives |
| **ROADWork** | Work-zone ground truth | Positive-rich (mostly 1) | Real work-zone structure, US coverage |
| **Open Images V7** | Global diversity | **ALL 1 (positives only)** | Diverse cones/barriers/signs |
| **Roboflow** | Curated work zones | ALL 1 | Fast high-density positives |
| **GTSRB Class 25** | EU roadwork signs | ALL 1 | EU sign patterns (NATIX is Europe-heavy) |
| **BDD100K (optional)** | Normal driving | Mainly 0 | Hard negatives, realism |

---

## 🚀 Complete Download Workflow (Laptop)

### Step 0: Verify NATIX
```bash
cd ~/projects/miner_b/streetvision_cascade

# Verify structure (NEVER modify this val set!)
ls data/natix_official/train/train_labels.csv
ls data/natix_official/val/val_labels.csv
```

---

### Step 1: ROADWork (Manual Download)

**Visit**: https://github.com/anuragxel/roadwork-dataset

**Follow their README** to find CMU KiltHub link.

**Download these ZIPs**:
- `images.zip`
- `annotations.zip`
- (Optional) `sem_seg_labels.zip`

**Unzip into correct structure**:
```bash
mkdir -p data/roadwork_iccv/raw
cd data/roadwork_iccv/raw

# Unzip following their directory structure
# Expected result:
# data/roadwork_iccv/raw/
#   scene/
#     images/
#     annotations/
#       instances_train.json
#       instances_val.json

cd ../../..

# Process with corrected parser
python3 prepare_roadwork_data.py --process_roadwork
```

**Expected Output**:
```
✅ TRAIN: ~4000-5000 samples
   Work zones: ~3500 (70-80%)
   Clean roads: ~500-1000 (20-30%)
```

---

### Step 2: Open Images V7 (Positives Only)

```bash
pip install fiftyone

# Run corrected script (treats all as positives)
python3 download_open_images_positives_only.py
```

**Expected Output**:
```
✅ Downloaded ~2000 images
   ALL labeled as 1 (positives)
⚠️  Remember: Your negatives come from NATIX!
```

---

### Step 3: Roboflow Work Zone

**Manual Download**:
1. Visit: https://universe.roboflow.com/workzone/roadwork
2. Click "Download Dataset" → Choose "COCO JSON"
3. Download ZIP to `~/Downloads/roboflow_roadwork.zip`

**Process**:
```bash
mkdir -p data/roadwork_extra/raw
unzip ~/Downloads/roboflow_roadwork.zip -d data/roadwork_extra/raw/

python3 prepare_roadwork_data.py --process_extra
```

**Expected Output**:
```
✅ ~500-1000 samples
   ALL labeled as 1 (focused dataset)
```

---

### Step 4: GTSRB Class 25 (EU Signs)

```bash
pip install kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb_class25/raw/

# Run corrected converter (properly converts .ppm to .png)
python3 convert_gtsrb_class25.py
```

**Expected Output**:
```
✅ Converted ~600 images to PNG
   ALL labeled as 1 (EU roadwork signs)
```

---

### Step 5: Verify All Datasets

```bash
# Check all CSVs exist
ls -lh data/*/train_labels.csv

# Quick stats
wc -l data/*/train_labels.csv

# Total expected: ~18,000-19,000 samples
# - NATIX: ~10,000
# - ROADWork: ~4,000-5,000
# - Open Images: ~2,000
# - Roboflow: ~500-1,000
# - GTSRB: ~600
```

---

## 🎓 Training Strategy (Corrected)

### Baseline (Establish Metrics)
```bash
python3 train_stage1_head.py --mode train --epochs 10
```
- **Train**: NATIX only (~10K)
- **Val**: NATIX only (~2.5K)
- **Expected**: 96-97% accuracy
- **Cost**: ~$0.90

### Aggressive (Max Positives)
```bash
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```
- **Train**: NATIX + ROADWork + Open Images + Roboflow + GTSRB (~18K)
- **Val**: NATIX only (~2.5K) ← **NEVER CHANGES**
- **Expected**: 97-98% accuracy (modest improvement)
- **Cost**: ~$1.50

**Why This Works**:
- NATIX provides balanced distribution + negatives
- External datasets add diverse positive examples
- Validation stays pure → metrics match deployment

---

## 📦 Transfer to SSH Server

```bash
# On laptop: Compress everything
cd ~/projects/miner_b
tar -czf miner_data.tar.gz \
  streetvision_cascade/data/natix_official \
  streetvision_cascade/data/roadwork_iccv \
  streetvision_cascade/data/open_images \
  streetvision_cascade/data/roadwork_extra \
  streetvision_cascade/data/gtsrb_class25 \
  models/

# Check size
ls -lh miner_data.tar.gz
# Expected: ~75-85 GB (60GB models + 15-25GB data)

# Transfer to SSH server
scp miner_data.tar.gz user@vast.ai:/workspace/

# On SSH server
ssh user@vast.ai
cd /workspace
tar -xzf miner_data.tar.gz

# Clone repo
git clone https://github.com/YOUR_USERNAME/miner_b.git
cd miner_b

# Move data/models into repo
mv /workspace/streetvision_cascade/data /workspace/miner_b/streetvision_cascade/
mv /workspace/models /workspace/miner_b/

# Install dependencies
pip install torch torchvision transformers timm datasets pillow tqdm scikit-learn

# Run baseline first
python3 streetvision_cascade/train_stage1_head.py --mode train --epochs 10

# Then aggressive mode
python3 streetvision_cascade/train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```

---

## 📈 Expected Results (Realistic)

### What to Expect
- **Val Accuracy**: +1-2% improvement (97-98% vs 96-97%)
- **US Work Zone Recall**: +5-10% (ROADWork adds US coverage)
- **EU Sign Recall**: +5-10% (GTSRB adds EU patterns)
- **ECE**: Slightly lower (better calibration)
- **Exit Accuracy @ 0.88**: More stable

### What NOT to Expect
- ❌ +32.5% precision boost (that's from ROADWork's discovery pipeline)
- ❌ Dramatic accuracy jumps (you already have good NATIX data)
- ❌ Perfect work zone detection (it's a hard problem)

### Why Modest Improvement is Still Pro
- Geographic robustness (EU + US)
- Better edge case handling
- Fewer false negatives on rare layouts
- More confident predictions (better ECE)

---

## ✅ Final Checklist (Before SSH)

On your laptop, verify:

- [ ] NATIX data verified (train/val split looks good)
- [ ] ROADWork downloaded from CMU KiltHub (ZIPs)
- [ ] ROADWork processed with corrected parser
- [ ] Open Images downloaded (positives only, ~2K)
- [ ] Roboflow downloaded and processed (~500-1K)
- [ ] GTSRB Class 25 converted properly (.ppm → .png)
- [ ] All `train_labels.csv` files exist
- [ ] Total samples: ~18,000-19,000
- [ ] `miner_data.tar.gz` created (~75-85 GB)
- [ ] Training scripts have syntax errors fixed

---

## 🔧 Files Created/Updated

### New Scripts (Corrected)
1. `download_open_images_positives_only.py` - Treats all as positives
2. `convert_gtsrb_class25.py` - Properly converts .ppm to .png
3. `prepare_roadwork_data.py` - Updated with real COCO parser

### Updated Scripts
1. `train_stage1_head.py` - Fixed syntax errors, added Open Images + GTSRB support

### Documentation
1. `DATASET_DOWNLOAD_CORRECTED.md` - This file (corrected guide)
2. `CORRECTED_DATASET_PLAN.md` - Summary of corrections

---

## 💡 Key Takeaways (Pro 2025 Approach)

1. **External datasets are positives boosters** - Don't expect balanced negatives
2. **NATIX is your anchor** - Provides negatives and deployment distribution
3. **Validation = NATIX only** - Never mix external datasets here
4. **Manual downloads required** - No magic scripts for ROADWork
5. **Modest improvements expected** - +1-2% accuracy, better robustness
6. **Geographic diversity matters** - EU (NATIX, GTSRB) + US (ROADWork) coverage

---

Last updated: 2025-12-23 (Final Corrected Version)
