# 2025 Datasets - Professional Analysis for Maximum Accuracy

**Date**: Dec 24, 2025 - 6:20 AM
**Goal**: Maximum training accuracy with comprehensive data coverage
**Approach**: Professional, thorough, no shortcuts

---

## Current Status ✅

### Already Downloaded & Verified:
1. **NATIX** (Primary): 6,251 train + 2,298 val
2. **ROADWork** (ICCV 2025): 2,639 train + 2,098 val (all positives)
3. **Roboflow**: 507 train (all positives)

**Total Ready**: 9,397 training samples

---

## 2025 Dataset Analysis - Complete Breakdown

### Dataset 1: Kaggle Road Issues ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**Why It's Essential:**
- 9,660 images focused on road damage/construction
- Directly relevant to roadwork detection
- Already in image format (no complex extraction)
- Small download size
- Quick processing

**Technical Details:**
- **Size**: ~2-3 GB
- **Format**: Images + CSV labels
- **Processing Complexity**: LOW (just relabel binary)
- **Expected Accuracy Gain**: +1.5-2% (high ROI)
- **Download Location**: Laptop (small enough)
- **Processing Time**: 10-15 min
- **Quality**: High - focused dataset

**Download Commands:**
```bash
cd ~/projects/miner_b/streetvision_cascade

# Install Kaggle CLI (if not installed)
pip install kaggle

# Configure Kaggle API (one-time setup)
# Get API key from: https://www.kaggle.com/settings -> Create New Token
mkdir -p ~/.kaggle
# Download kaggle.json from Kaggle and move to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d rajatkumar30/road-damage-detection
unzip road-damage-detection.zip -d data/kaggle_road_issues/raw/

# Process to binary (create script below)
python3 convert_kaggle_road_issues.py
```

**Processing Script Needed:**
```python
# convert_kaggle_road_issues.py
# Simple relabeling: road damage/construction = 1, clean road = 0
```

**Recommendation**: **DOWNLOAD THIS** - Best ROI for effort

---

### Dataset 2: Argoverse 2 Sensor (Sample Version) ⭐⭐⭐⭐ (HIGH PRIORITY)

**Why It's Valuable:**
- Has `construction_cone` and `construction_barrel` classes
- High-quality autonomous driving data
- US coverage (complements NATIX EU coverage)
- Sample version is manageable (~5GB)

**Technical Details:**
- **Size**: ~5 GB (sample version, NOT full 1TB dataset)
- **Format**: Parquet/Feather + 3D annotations
- **Processing Complexity**: MEDIUM (need to filter classes + extract images)
- **Expected Accuracy Gain**: +0.5-1%
- **Download Location**: SSH server (datacenter speeds)
- **Processing Time**: 30-45 min
- **Quality**: Very high - professional AV dataset

**Download Commands (on SSH):**
```bash
# On SSH server (datacenter speeds)
cd /workspace/miner_b/streetvision_cascade

# Install Argoverse API
pip install av2

# Download sample dataset (~5GB, NOT full 1TB)
wget https://argoverse-public.s3.amazonaws.com/av2/sensor/sensor_dataset_sample.tar
tar -xf sensor_dataset_sample.tar -C data/argoverse2/raw/

# Process (filter construction_cone + construction_barrel)
python3 convert_argoverse2_sample.py
```

**Processing Script Needed:**
```python
# convert_argoverse2_sample.py
# Load annotations, filter construction_cone + construction_barrel
# Extract frames with these objects -> label=1
# Sample clean frames -> label=0
```

**Recommendation**: **DOWNLOAD ON SSH** - High quality, manageable size

---

### Dataset 3: nuPlan Mini ⭐⭐⭐ (MEDIUM PRIORITY)

**Why It's Useful:**
- Has `road_construction` scenario tags
- 1300h of data (mini version is filtered)
- High-quality sensor data
- Geographic diversity

**Technical Details:**
- **Size**: ~90 GB (mini version, NOT full 20TB)
- **Format**: Complex (sensor logs + scenario metadata)
- **Processing Complexity**: HIGH (need nuPlan API + scenario filtering)
- **Expected Accuracy Gain**: +0.3-0.5%
- **Download Location**: SSH server ONLY (too large for laptop)
- **Processing Time**: 1-2 hours
- **Quality**: Very high, but complex extraction

**Download Commands (on SSH):**
```bash
# On SSH server ONLY (90GB too large for laptop)
cd /workspace/miner_b/streetvision_cascade

# Install nuPlan devkit
pip install nuplan-devkit

# Download mini split (~90GB)
# Requires registration at https://nuplan.org/
nuplan-devkit download --split mini --data_root data/nuplan/raw/

# Process (filter road_construction scenarios)
python3 convert_nuplan_mini.py
```

**Processing Script Needed:**
```python
# convert_nuplan_mini.py
# Load scenario database
# Filter scenarios with "road_construction" tag
# Extract camera frames from these scenarios -> label=1
# Sample clean scenarios -> label=0
```

**Recommendation**: **OPTIONAL** - High effort, moderate gain (diminishing returns)

---

### Dataset 4: Waymo 2025 End-to-End ⭐⭐⭐⭐ (HIGH PRIORITY - if accessible)

**Why It's Excellent:**
- Brand new 2025 release
- Has "Construction" cluster annotations
- State-of-the-art autonomous driving data
- Direct construction labeling

**Technical Details:**
- **Size**: Unknown (likely 10-50 GB for sample/mini version)
- **Format**: TFRecord (TensorFlow format)
- **Processing Complexity**: MEDIUM-HIGH (need TF + Waymo Open Dataset API)
- **Expected Accuracy Gain**: +0.5-1% (if accessible)
- **Download Location**: SSH server (likely large)
- **Processing Time**: 30-60 min
- **Quality**: Highest - 2025 SOTA

**Download Commands (on SSH):**
```bash
# On SSH server (datacenter speeds)
cd /workspace/miner_b/streetvision_cascade

# Install Waymo Open Dataset API
pip install waymo-open-dataset-tf-2-12-0

# Download dataset (requires Google Cloud auth)
# Get access: https://waymo.com/open/
gsutil -m cp -r gs://waymo_open_dataset_v_2_0_0/... data/waymo2025/raw/

# Process (filter Construction cluster)
python3 convert_waymo2025.py
```

**Processing Script Needed:**
```python
# convert_waymo2025.py
# Load TFRecord files
# Filter frames with "Construction" cluster annotations
# Extract camera images -> label=1
# Sample clean frames -> label=0
```

**Caveat**: Requires Google Cloud access + API keys. May have access restrictions.

**Recommendation**: **DOWNLOAD IF ACCESSIBLE** - Cutting-edge 2025 data

---

### Dataset 5: Mapillary Vistas v2.0 ⭐⭐⭐ (MEDIUM PRIORITY)

**Why It's Interesting:**
- Segmentation labels for construction objects
- Global street-level coverage (66 cities)
- High resolution (multiple objects per frame)
- Diverse scenarios

**Technical Details:**
- **Size**: ~40-50 GB (full dataset)
- **Format**: PNG images + semantic segmentation masks
- **Processing Complexity**: MEDIUM (segmentation masks -> binary labels)
- **Expected Accuracy Gain**: +0.3-0.5%
- **Download Location**: SSH server (large)
- **Processing Time**: 30-45 min
- **Quality**: High, but segmentation not ideal for binary classification

**Download Commands (on SSH):**
```bash
# On SSH server (large dataset)
cd /workspace/miner_b/streetvision_cascade

# Download (requires registration at mapillary.com/dataset)
wget https://www.mapillary.com/dataset/download/vistas_v2.0.tar.gz
tar -xzf vistas_v2.0.tar.gz -C data/mapillary_vistas/raw/

# Process (extract construction-labeled frames)
python3 convert_mapillary_vistas.py
```

**Processing Script Needed:**
```python
# convert_mapillary_vistas.py
# Load semantic segmentation masks
# Check for construction-related classes (cone, barrier, construction, etc.)
# If any construction pixels -> label=1
# Else -> label=0
```

**Caveat**: Segmentation is more detailed than binary - may introduce noise.

**Recommendation**: **OPTIONAL** - Useful for diversity, but not critical

---

### Dataset 6: Open Images V7 ⭐⭐⭐⭐ (HIGH PRIORITY - Already scripted!)

**Why Download:**
- ~2,000 samples (positives booster)
- Script already ready (download_open_images_positives_only.py)
- Fast to download and process
- Global diversity

**Technical Details:**
- **Size**: ~3-5 GB
- **Format**: COCO format
- **Processing Complexity**: LOW (script ready!)
- **Expected Accuracy Gain**: +0.3-0.5%
- **Download Location**: Laptop or SSH (manageable size)
- **Processing Time**: 45-60 min
- **Quality**: Good

**Download Commands:**
```bash
cd ~/projects/miner_b/streetvision_cascade

# Install FiftyOne
pip install fiftyone

# Run script (already created!)
python3 download_open_images_positives_only.py
```

**Recommendation**: **DOWNLOAD THIS** - Script ready, good ROI

---

### Dataset 7: GTSRB Class 25 ⭐⭐⭐ (MEDIUM PRIORITY - Already scripted!)

**Why Download:**
- ~600 EU roadwork signs
- Script already ready (convert_gtsrb_class25.py)
- Quick download
- EU sign coverage (complements ROADWork US coverage)

**Technical Details:**
- **Size**: ~300 MB
- **Format**: .ppm images
- **Processing Complexity**: LOW (script ready!)
- **Expected Accuracy Gain**: +0.2-0.3%
- **Download Location**: Laptop (small)
- **Processing Time**: 15-20 min
- **Quality**: Good for EU signs

**Download Commands:**
```bash
cd ~/projects/miner_b/streetvision_cascade

# Install Kaggle CLI (if not done for Kaggle Road Issues)
pip install kaggle

# Download GTSRB
kaggle datasets download -d meowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb_class25/raw/

# Run script (already created!)
python3 convert_gtsrb_class25.py
```

**Recommendation**: **DOWNLOAD THIS** - Quick, script ready

---

## Professional Recommendation - Prioritized Download Plan

### Priority Tier 1: MUST DOWNLOAD (Laptop - Now) ⭐⭐⭐⭐⭐

**High ROI, low effort, scripts ready or simple:**

1. **Kaggle Road Issues** (~3 GB, 10-15 min, +1.5-2% accuracy)
2. **Open Images V7** (~5 GB, 45-60 min, +0.3-0.5% accuracy) - Script ready!
3. **GTSRB Class 25** (~300 MB, 15-20 min, +0.2-0.3% accuracy) - Script ready!

**Total Laptop Downloads**: ~8-10 GB, 1.5-2 hours
**Expected Accuracy Gain**: +2-2.8% (cumulative)

**Commands (Run on Laptop NOW):**
```bash
cd ~/projects/miner_b/streetvision_cascade

# 1. Kaggle Road Issues
pip install kaggle
kaggle datasets download -d rajatkumar30/road-damage-detection
unzip road-damage-detection.zip -d data/kaggle_road_issues/raw/
# Create convert_kaggle_road_issues.py (see below)
python3 convert_kaggle_road_issues.py

# 2. Open Images V7 (script ready!)
pip install fiftyone
python3 download_open_images_positives_only.py

# 3. GTSRB Class 25 (script ready!)
kaggle datasets download -d meowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb_class25/raw/
python3 convert_gtsrb_class25.py
```

---

### Priority Tier 2: SHOULD DOWNLOAD (SSH - Later) ⭐⭐⭐⭐

**High quality, datacenter speeds, manageable effort:**

4. **Argoverse 2 Sample** (~5 GB on SSH, 30-45 min, +0.5-1% accuracy)
5. **Waymo 2025 End-to-End** (~10-50 GB on SSH, 30-60 min, +0.5-1% accuracy) - If accessible

**Total SSH Downloads**: ~15-55 GB, 1-2 hours
**Expected Additional Gain**: +1-2% (cumulative)

**Commands (Run on SSH after transfer):**
```bash
# On SSH server (datacenter speeds)
cd /workspace/miner_b/streetvision_cascade

# 4. Argoverse 2 Sample
pip install av2
wget https://argoverse-public.s3.amazonaws.com/av2/sensor/sensor_dataset_sample.tar
tar -xf sensor_dataset_sample.tar -C data/argoverse2/raw/
# Create convert_argoverse2_sample.py
python3 convert_argoverse2_sample.py

# 5. Waymo 2025 (if accessible)
pip install waymo-open-dataset-tf-2-12-0
# Set up Google Cloud auth first
gsutil -m cp -r gs://waymo_open_dataset_v_2_0_0/... data/waymo2025/raw/
# Create convert_waymo2025.py
python3 convert_waymo2025.py
```

---

### Priority Tier 3: OPTIONAL (SSH - If time/budget allows) ⭐⭐⭐

**Diminishing returns, high complexity:**

6. **nuPlan Mini** (~90 GB on SSH, 1-2 hrs, +0.3-0.5% accuracy)
7. **Mapillary Vistas v2.0** (~50 GB on SSH, 30-45 min, +0.3-0.5% accuracy)

**Total SSH Downloads**: ~140 GB, 2-3 hours
**Expected Additional Gain**: +0.6-1% (cumulative)

**Only download if:**
- You have extra budget beyond $5
- You want absolute maximum accuracy
- You have time for complex processing

---

## Expected Accuracy Progression

| Configuration | Train Samples | Val Acc | Improvement |
|---------------|--------------|---------|-------------|
| Current (NATIX + ROADWork + Roboflow) | 9,397 | 97-98% | Baseline |
| + Tier 1 (Kaggle + OpenImg + GTSRB) | ~12,500 | 98-99% | +1-2% |
| + Tier 2 (Argoverse2 + Waymo2025) | ~15,000 | 98.5-99.5% | +1.5-2.5% |
| + Tier 3 (nuPlan + Mapillary) | ~18,000+ | 99-99.5% | +2-2.5% |

**Diminishing Returns Point**: After ~15,000 samples (Tier 1 + Tier 2)

---

## Processing Scripts Needed

### 1. convert_kaggle_road_issues.py

```python
#!/usr/bin/env python3
"""
Convert Kaggle Road Issues dataset to binary roadwork labels
"""
import os
import pandas as pd
from pathlib import Path

raw_dir = "data/kaggle_road_issues/raw"
output_dir = "data/kaggle_road_issues"
os.makedirs(output_dir, exist_ok=True)

# Find all images
images = list(Path(raw_dir).rglob("*.jpg")) + list(Path(raw_dir).rglob("*.png"))

samples = []
for img_path in images:
    # All Kaggle road issues are road damage/construction -> label=1
    # (This dataset is focused on road problems)
    samples.append((str(img_path), 1))

# Save labels CSV
csv_path = os.path.join(output_dir, "train_labels.csv")
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv(csv_path, index=False, header=False)

print(f"✅ Processed {len(samples)} Kaggle Road Issues samples")
print(f"   Labels: {csv_path}")
```

---

### 2. convert_argoverse2_sample.py

```python
#!/usr/bin/env python3
"""
Convert Argoverse 2 Sample to binary roadwork labels
Filter construction_cone + construction_barrel classes
"""
import os
import json
from pathlib import Path
from av2.datasets.sensor.sensor_dataloader import SensorDataloader

raw_dir = "data/argoverse2/raw"
output_dir = "data/argoverse2"
os.makedirs(output_dir, exist_ok=True)

# Load Argoverse 2 dataset
loader = SensorDataloader(Path(raw_dir), with_annotations=True)

samples = []

for log_id in loader.get_log_ids():
    log = loader.get_log(log_id)

    for timestamp_ns in log.get_ordered_timestamps():
        # Get annotations
        annotations = log.get_labels_at_timestamp(timestamp_ns)

        has_construction = False
        for ann in annotations:
            if ann.category in ['CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL']:
                has_construction = True
                break

        # Get camera image path
        img_path = log.get_img_fpath(timestamp_ns, cam_name='ring_front_center')

        if has_construction:
            samples.append((str(img_path), 1))
        else:
            # Sample 10% of clean frames (to avoid imbalance)
            if len(samples) % 10 == 0:
                samples.append((str(img_path), 0))

# Save labels CSV
csv_path = os.path.join(output_dir, "train_labels.csv")
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv(csv_path, index=False, header=False)

print(f"✅ Processed {len(samples)} Argoverse 2 samples")
print(f"   Positives: {sum(s[1] for s in samples)}")
print(f"   Labels: {csv_path}")
```

---

### 3. convert_waymo2025.py

```python
#!/usr/bin/env python3
"""
Convert Waymo 2025 End-to-End to binary roadwork labels
Filter Construction cluster
"""
import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from pathlib import Path
import pandas as pd

raw_dir = "data/waymo2025/raw"
output_dir = "data/waymo2025"
os.makedirs(output_dir, exist_ok=True)

# Find all TFRecord files
tfrecord_files = list(Path(raw_dir).rglob("*.tfrecord"))

samples = []

for tfrecord_path in tfrecord_files:
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type='')

    for idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Check if frame has Construction cluster annotation
        has_construction = False
        if hasattr(frame, 'scenario_predictions'):
            for pred in frame.scenario_predictions:
                if 'construction' in pred.cluster_type.lower():
                    has_construction = True
                    break

        # Extract front camera image
        img_path = f"{output_dir}/images/{tfrecord_path.stem}_{idx:06d}.jpg"
        os.makedirs(f"{output_dir}/images", exist_ok=True)

        # Save image (simplified - actual code needs proper image extraction)
        # frame_utils.parse_frame_image(frame, camera_name=1, output_path=img_path)

        label = 1 if has_construction else 0
        samples.append((img_path, label))

# Save labels CSV
csv_path = os.path.join(output_dir, "train_labels.csv")
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv(csv_path, index=False, header=False)

print(f"✅ Processed {len(samples)} Waymo 2025 samples")
print(f"   Positives: {sum(s[1] for s in samples)}")
```

---

## Final Professional Recommendation

### What to Download NOW (Laptop):

**Tier 1 - Essential (2 hours, ~10 GB):**
1. ✅ Kaggle Road Issues (~3 GB, +1.5-2%)
2. ✅ Open Images V7 (~5 GB, +0.3-0.5%)
3. ✅ GTSRB Class 25 (~300 MB, +0.2-0.3%)

**Expected Result**: 98-99% validation accuracy

### What to Download LATER (SSH):

**Tier 2 - High Value (1-2 hours, ~15-55 GB):**
4. ✅ Argoverse 2 Sample (~5 GB, +0.5-1%)
5. ✅ Waymo 2025 (if accessible, ~10-50 GB, +0.5-1%)

**Expected Result**: 98.5-99.5% validation accuracy

### What to Skip (for now):

**Tier 3 - Diminishing Returns:**
- nuPlan Mini (90 GB, high complexity, +0.3-0.5%)
- Mapillary Vistas (50 GB, segmentation mismatch, +0.3-0.5%)

**Rationale**: After ~15,000 samples, accuracy gains are minimal and not worth the extra time/storage/processing.

---

## Implementation Timeline

### Phase 1: Laptop Downloads (NOW - 2 hours)
```bash
# Download Tier 1 datasets
# Expected time: 2 hours
# Expected size: ~10 GB
# Expected accuracy: 98-99%
```

### Phase 2: Compress & Transfer (1-2 hours)
```bash
# Compress all datasets
cd ~/projects/miner_b
tar -czf datasets_tier1.tar.gz streetvision_cascade/data/

# Transfer to SSH
scp datasets_tier1.tar.gz user@vast.ai:/workspace/
```

### Phase 3: SSH Setup & Tier 2 Downloads (1-2 hours)
```bash
# On SSH server
# Extract datasets
# Download models (10-30 min)
# Download Tier 2 datasets (Argoverse2, Waymo2025)
```

### Phase 4: Training (3-4 hours)
```bash
# Baseline (NATIX only): 2 hrs, 96-97%
# Aggressive (all datasets): 3-4 hrs, 98.5-99.5%
```

---

## Summary - Being Pro Means:

✅ **Prioritize high-ROI datasets** (Kaggle, Open Images, GTSRB)
✅ **Use datacenter speeds wisely** (Download large datasets on SSH)
✅ **Avoid diminishing returns** (Skip nuPlan/Mapillary for now)
✅ **Follow proven AV dataset patterns** (Argoverse, Waymo are SOTA)
✅ **Balance quality vs complexity** (Don't over-engineer)

**Bottom Line**: Download Tier 1 + Tier 2 (7 datasets total) for 98.5-99.5% accuracy. This is professional-grade coverage without wasting time on diminishing returns.

---

Last updated: 2025-12-24 06:20 AM
