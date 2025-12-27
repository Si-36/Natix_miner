# 🚀 SSH Quick Start Guide

## ⚡ TL;DR - Run This on SSH

```bash
# SSH into server
ssh ubuntu@your-server

# Navigate to project
cd ~/Natix_miner/streetvision_cascade

# Activate venv
source .venv/bin/activate

# Run automated download script
bash scripts/download_all_datasets_ssh.sh
```

---

## 📁 Why Two Roadwork Folders?

| Folder | What It Is | Why You Need It |
|--------|-----------|-----------------|
| **`roadwork_iccv/`** | CMU ROADWork ICCV 2025<br>~5,000 US work zones | **Best quality** - State-of-the-art benchmark<br>32.5% precision boost in paper<br>Diverse conditions (night, rain, etc.) |
| **`roadwork_extra/`** | Roboflow Universe<br>~500 curated images | **Quick extras** - Additional edge cases<br>Different camera types<br>Fast to download |

**Answer**: Both are correct! They serve different purposes:
- `roadwork_iccv/` = High-quality, research-grade dataset
- `roadwork_extra/` = Quick curated positives

---

## 🎯 Complete Dataset Strategy

### **Primary (Must Have)**
1. **NATIX Official** - `data/natix_official/`
   - Source: Hugging Face `natix-network-org/roadwork`
   - Size: ~6K-10K train, ~2K-2.5K val
   - **This is your deployment distribution**

### **Best External (Highly Recommended)**
2. **ROADWork ICCV** - `data/roadwork_iccv/`
   - Source: CMU GitHub (manual download)
   - Size: ~4.5K-5K train
   - **32.5% precision boost proven**

### **Quick Extras (Optional)**
3. **Roboflow** - `data/roadwork_extra/`
   - Source: Roboflow Universe
   - Size: ~500 images
   - **Fast to add**

4. **Open Images** - `data/open_images/`
   - Source: Open Images V7
   - Size: ~2K positives
   - **Global diversity**

5. **GTSRB Class 25** - `data/gtsrb_class25/`
   - Source: GTSRB official
   - Size: ~1.5K EU signs
   - **European coverage**

6. **Kaggle Road Issues** - `data/kaggle_road_issues/`
   - Source: Kaggle
   - Size: ~6K negatives
   - **Hard negatives (road problems ≠ roadwork)**

---

## 📥 Download Commands (SSH)

### **1. NATIX (Automatic)**
```bash
python3 << 'PY'
from datasets import load_dataset
dataset = load_dataset("natix-network-org/roadwork", split="train")
dataset.to_parquet("data/natix_official/data/train.parquet")
PY

python3 convert_natix_parquet.py
```

### **2. ROADWork (Manual - Best External)**
```bash
# Step 1: Visit GitHub
# https://github.com/anuragxel/roadwork-dataset
# Follow their download instructions (CMU KiltHub or Google Drive)

# Step 2: Extract to:
# data/roadwork_iccv/raw/images/
# data/roadwork_iccv/raw/annotations/

# Step 3: Process
python3 prepare_roadwork_data.py --process_roadwork
```

### **3. Roboflow (Manual - Quick Extra)**
```bash
# Step 1: Visit
# https://universe.roboflow.com/workzone/roadwork
# Download COCO JSON format

# Step 2: Extract to:
# data/roadwork_extra/raw/

# Step 3: Process
python3 prepare_roadwork_data.py --process_extra
```

### **4. Open Images (Automatic)**
```bash
python3 download_open_images_positives_only_no_mongo.py
```

### **5. GTSRB (Automatic)**
```bash
mkdir -p data/gtsrb_class25/raw
cd data/gtsrb_class25/raw
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip
cd ../../..
python3 convert_gtsrb_class25.py
```

### **6. Kaggle (Manual)**
```bash
# Download from Kaggle, extract to:
# data/kaggle_road_issues/raw/

python3 convert_kaggle_road_issues.py
```

---

## ✅ Verify Everything

```bash
python3 verify_datasets.py --check_all
```

**Expected Output**:
```
✅ NATIX Official: 6,251 train, 2,298 val
✅ ROADWork ICCV: 4,523 train, 1,201 val
✅ Roboflow Extra: 847 train
✅ Open Images: 2,000 train
✅ GTSRB Class 25: 1,500 train
✅ Kaggle Road Issues: 6,113 train
```

---

## 🏋️ Training Modes

### **Baseline (NATIX Only)**
```bash
python3 train_stage1_head.py --mode train --epochs 10
```
- Uses: ~6K-10K samples
- Fast iteration

### **Aggressive (All Datasets)**
```bash
python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork
```
- Uses: ~19K-25K samples
- Maximum accuracy

---

## 🔍 ROADWork Deep Dive

### **What Makes ROADWork Special**

1. **ICCV 2025 Accepted** - Peer-reviewed benchmark
2. **18 US Cities** - Geographic diversity
3. **Rich Annotations** - Cones, barriers, signs, workers
4. **Diverse Conditions** - Day, night, rain, snow
5. **Proven Results** - 32.5% precision improvement

### **ROADWork Structure**

```
data/roadwork_iccv/
├── raw/                    # Original download
│   ├── images/            # ~8,500 images (JPG)
│   ├── annotations/        # COCO JSON files
│   └── sem_seg_labels/     # Segmentation masks
├── train_labels.csv        # Binary: image_path,label
└── val_labels.csv         # Binary: image_path,label
```

### **ROADWork Label Logic**

```python
# Any work zone object = positive
label = 1 if (
    cones > 0 OR
    barriers > 0 OR
    arrow_boards > 0 OR
    work_zone_signs > 0 OR
    workers > 0
) else 0
```

---

## 🐛 Common Issues

### **ROADWork Download Link Not Found**
- Check GitHub README: https://github.com/anuragxel/roadwork-dataset
- May require CMU KiltHub access (academic email)
- Alternative: Use Roboflow + Open Images (still good)

### **NATIX Download Slow**
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 convert_natix_parquet.py
```

### **Processing Fails**
- Check file structure: `ls -R data/roadwork_iccv/raw/`
- Adjust parser if format differs (COCO vs Pascal VOC vs YOLO)

---

## 📊 Final Checklist

- [ ] NATIX downloaded and converted
- [ ] ROADWork downloaded and processed (if available)
- [ ] Roboflow downloaded and processed (optional)
- [ ] Open Images downloaded
- [ ] GTSRB downloaded and converted
- [ ] Kaggle downloaded and converted (optional)
- [ ] All datasets verified: `python3 verify_datasets.py --check_all`
- [ ] Baseline training works: `python3 train_stage1_head.py --mode train --epochs 1`
- [ ] Aggressive training works: `python3 train_stage1_head.py --mode train --epochs 1 --use_extra_roadwork`

---

## 🎯 Key Takeaways

1. ✅ **Two roadwork folders are correct** - Different purposes
2. ✅ **Download on SSH** - Faster than 34GB transfer
3. ✅ **ROADWork is critical** - 32.5% boost proven
4. ✅ **Always validate on NATIX val** - Keep metric pure
5. ✅ **Combined = ~19K-25K samples** - Maximum accuracy

---

**Last Updated**: 2025-12-24
**Status**: Ready for SSH deployment

