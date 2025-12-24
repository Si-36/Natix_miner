# Laptop Setup Guide - Step by Step

**Goal**: Download and prepare all datasets on your laptop, then upload ONLY datasets to SSH server.

**Time**: ~2-3 hours (mostly download time)

---

## 🚀 START HERE - Step by Step

### ✅ Step 1: Verify NATIX Dataset (5 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Run verification
python3 verify_datasets.py --check_natix
```

**Expected Output**:
```
✅ NATIX dataset OK!
   Train: ~10000 samples
   Val:   ~2500 samples
```

**If errors**: Fix NATIX paths before continuing.

**When done**: Tell me "Step 1 done" and I'll mark it complete.

---

### 📥 Step 2: Download ROADWork (30 min)

**Manual Download Required**:

1. Open browser: https://github.com/anuragxel/roadwork-dataset
2. Find the CMU KiltHub link in their README
3. Download these files:
   - `images.zip` (largest file, ~3-4 GB)
   - `annotations.zip` (~50-100 MB)
4. Save to `~/Downloads/`

**When done**: Tell me "Step 2 done - downloaded images.zip and annotations.zip"

---

### 📦 Step 3: Unzip ROADWork (5 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Create directory
mkdir -p data/roadwork_iccv/raw

# Move downloaded files
mv ~/Downloads/images.zip data/roadwork_iccv/raw/
mv ~/Downloads/annotations.zip data/roadwork_iccv/raw/

# Unzip
cd data/roadwork_iccv/raw
unzip images.zip
unzip annotations.zip
cd ../../..
```

**Check it worked**:
```bash
ls data/roadwork_iccv/raw/
# Should see: images/ and annotations/ folders (or similar)
```

**When done**: Tell me "Step 3 done - unzipped ROADWork"

---

### ⚙️ Step 4: Process ROADWork to Binary Labels (5 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Process to binary labels
python3 prepare_roadwork_data.py --process_roadwork
```

**Expected Output**:
```
✅ TRAIN: ~4000-5000 samples
   Work zones: ~3500 (70-80%)
   Clean roads: ~500-1000 (20-30%)
```

**Check it worked**:
```bash
ls data/roadwork_iccv/train_labels.csv
# Should exist
head -5 data/roadwork_iccv/train_labels.csv
# Should show: path,label format
```

**If errors**: The script may need adjustment for actual ROADWork format. Show me the error.

**When done**: Tell me "Step 4 done - processed ROADWork"

---

### 🌍 Step 5: Install FiftyOne (5 min)

```bash
pip install fiftyone
```

**Check it worked**:
```bash
python3 -c "import fiftyone as fo; print(fo.__version__)"
# Should print version number
```

**When done**: Tell me "Step 5 done - FiftyOne installed"

---

### 🌎 Step 6: Download Open Images V7 (45 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Run the corrected download script
python3 download_open_images_positives_only.py
```

**What happens**:
- Downloads ~2000 images (this takes time!)
- Progress bar will show download status
- All images labeled as 1 (positives only)

**Expected Output**:
```
✅ Downloaded ~2000 images
   ALL labeled as 1 (positives)
⚠️  Remember: Your negatives come from NATIX!
```

**Check it worked**:
```bash
ls data/open_images/train_labels.csv
ls data/open_images/coco/data/ | head -5
# Should see image files
```

**When done**: Tell me "Step 6 done - downloaded Open Images V7"

---

### 🔥 Step 7: Download Roboflow (10 min)

**Manual Download Required**:

1. Open browser: https://universe.roboflow.com/workzone/roadwork
2. Click "Download Dataset"
3. Choose format: **COCO JSON**
4. Download ZIP to `~/Downloads/roboflow_roadwork.zip`

**When done**: Tell me "Step 7 done - downloaded Roboflow ZIP"

---

### 📦 Step 8: Unzip Roboflow (2 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Create directory
mkdir -p data/roadwork_extra/raw

# Unzip
unzip ~/Downloads/roboflow_roadwork.zip -d data/roadwork_extra/raw/
```

**Check it worked**:
```bash
ls data/roadwork_extra/raw/
# Should see images and annotation files
```

**When done**: Tell me "Step 8 done - unzipped Roboflow"

---

### ⚙️ Step 9: Process Roboflow (2 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Process to binary labels
python3 prepare_roadwork_data.py --process_extra
```

**Expected Output**:
```
✅ ~500-1000 samples
   ALL labeled as 1 (focused dataset)
```

**Check it worked**:
```bash
ls data/roadwork_extra/train_labels.csv
wc -l data/roadwork_extra/train_labels.csv
# Should show count of samples
```

**When done**: Tell me "Step 9 done - processed Roboflow"

---

### 🇪🇺 Step 10: Install Kaggle CLI (2 min)

```bash
pip install kaggle pillow
```

**Setup Kaggle API** (if first time):
1. Go to: https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save `kaggle.json` to `~/.kaggle/`
5. Run: `chmod 600 ~/.kaggle/kaggle.json`

**Check it worked**:
```bash
kaggle --version
# Should print version
```

**When done**: Tell me "Step 10 done - Kaggle CLI installed"

---

### 📥 Step 11: Download GTSRB from Kaggle (15 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Download from Kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
```

**Expected Output**:
```
Downloading gtsrb-german-traffic-sign.zip to ...
100%|████████████████████| ...
```

**Check it worked**:
```bash
ls gtsrb-german-traffic-sign.zip
# Should exist
```

**When done**: Tell me "Step 11 done - downloaded GTSRB"

---

### 📦 Step 12: Unzip GTSRB (5 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Unzip
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb_class25/raw/
```

**Check it worked**:
```bash
ls data/gtsrb_class25/raw/Train/25/
# Should see .ppm image files
```

**When done**: Tell me "Step 12 done - unzipped GTSRB"

---

### ⚙️ Step 13: Convert GTSRB Class 25 (5 min)

```bash
cd ~/projects/miner_b/streetvision_cascade

# Convert .ppm to .png (Class 25 only)
python3 convert_gtsrb_class25.py
```

**Expected Output**:
```
✅ Converted ~600 images to PNG
   ALL labeled as 1 (EU roadwork signs)
```

**Check it worked**:
```bash
ls data/gtsrb_class25/train_labels.csv
ls data/gtsrb_class25/train_images/*.png | wc -l
# Should show ~600 images
```

**When done**: Tell me "Step 13 done - converted GTSRB"

---

### ✅ Step 14: Verify ALL Datasets (5 min)

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
```

**If ANY errors**: Fix them before proceeding.

**When done**: Tell me "Step 14 done - all datasets verified!"

---

### 📦 Step 15: Compress ONLY Datasets (15-30 min)

**IMPORTANT**: We're compressing ONLY datasets (NOT models!)

```bash
cd ~/projects/miner_b

# Compress ONLY data (NOT models!)
tar -czf datasets_only.tar.gz streetvision_cascade/data/

# Check size
ls -lh datasets_only.tar.gz
```

**Expected Size**: ~10-20 GB (NOT 70-80 GB!)

**If size is >30 GB**: You included models by mistake. Delete and redo.

**When done**: Tell me "Step 15 done - compressed datasets"

---

### 📤 Step 16: Transfer to SSH Server (1-3 hours)

**Get your SSH server details** from Vast.ai:
- IP address
- Username (usually `root`)
- Port (if not 22)

**Transfer command**:
```bash
# Option A: scp
scp datasets_only.tar.gz user@vast.ai:/workspace/

# Option B: rsync (can resume if interrupted)
rsync -avz --progress datasets_only.tar.gz user@vast.ai:/workspace/
```

**Time estimate** (for 10-20 GB):
- 10 Mbps upload: ~2-3 hours
- 50 Mbps upload: ~30-45 min
- 100 Mbps upload: ~15-20 min

**Tip**: Run in `screen` or `tmux` so it continues if connection drops:
```bash
screen -S upload
scp datasets_only.tar.gz user@vast.ai:/workspace/
# Press Ctrl+A then D to detach
# Later: screen -r upload to reattach
```

**When done**: Tell me "Step 16 done - transferred to SSH"

---

## ✅ Laptop Setup Complete!

When all 16 steps are done, you're ready for SSH setup. I'll give you the SSH instructions then.

---

## 🛟 Troubleshooting

### ROADWork processing fails
**Cause**: Actual ROADWork format may differ from assumptions.
**Fix**: Show me the error and first few lines of the annotation JSON.

### Open Images download stuck
**Cause**: Large download, may take time.
**Fix**: Wait patiently. FiftyOne shows progress bar.

### GTSRB conversion fails
**Cause**: Missing PIL or wrong paths.
**Fix**: Check `data/gtsrb_class25/raw/Train/25/` has .ppm files.

### SCP transfer very slow
**Cause**: Your home upload speed is the bottleneck.
**Fix**: Let it run overnight, or use faster internet (coffee shop, university).

---

**Start with Step 1 and tell me when each step is done!** 🚀
