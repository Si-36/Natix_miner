# 🎯 SOLUTION: Get Images for Validator Testing

**Date:** December 19, 2025  
**Goal:** Populate validator's image cache so it can query your miner

---

## 🔍 What I Found

### NATIX Has 3 Ways to Get Images:

1. ✅ **Download from Hugging Face** (BEST for testing!)
   - NATIX has a public dataset: `natix-network-org/roadwork`
   - Contains real roadwork images
   - Free to download
   - Already configured in the code!

2. ✅ **Use Sample Image** (Quick test)
   - NATIX includes a test image in their repo
   - Location: `neurons/unit_tests/sample_image.jpg`
   - We can copy this to cache

3. ⚠️ **Generate Synthetic Images** (Complex, needs GPU + setup)
   - Requires Stable Diffusion models
   - Needs Hugging Face token
   - More complex setup

---

## 🚀 RECOMMENDED SOLUTION

### Option 1: Quick Test with Sample Image (2 minutes)

Let's use NATIX's own sample image to test immediately:

```bash
# Create cache directory
mkdir -p ~/.cache/natix/Roadwork/images/

# Copy NATIX's sample image
cp /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet/neurons/unit_tests/sample_image.jpg \
   ~/.cache/natix/Roadwork/images/image_001.jpg

# Verify it's there
ls -lh ~/.cache/natix/Roadwork/images/

# Restart validator to pick up the image
pkill -f "neurons/validator.py"
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &

# Watch logs to see it work!
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log
```

**Expected result:**
- Validator will find the image
- Send it to your miner (UID 88)
- Miner will analyze and respond
- You'll see actual query/response in logs!

---

### Option 2: Download Real Dataset from Hugging Face (10 minutes)

NATIX has a cache updater that automatically downloads images from their Hugging Face dataset!

#### Step 1: Run the Cache Updater

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# This will download real roadwork images from Hugging Face
./start_cache_updater.sh > /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log 2>&1 &

# Watch it download images
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log
```

#### What the Cache Updater Does:

1. **Downloads parquet files** from `natix-network-org/roadwork` dataset on Hugging Face
2. **Extracts 100 images** from each parquet file
3. **Saves images** to `~/.cache/natix/Roadwork/images/`
4. **Updates every hour** to get fresh images

#### Step 2: Wait for Download (5-10 minutes)

The cache updater will:
- Download parquet files (compressed image data)
- Extract images from parquet files
- Save to `~/.cache/natix/Roadwork/images/`

Check progress:
```bash
# Watch cache updater logs
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log

# Check if images are appearing
watch -n 5 "ls -lh ~/.cache/natix/Roadwork/images/ | wc -l"
```

#### Step 3: Validator Will Automatically Use Images

Once images appear in cache, your validator will automatically:
- Pick random images
- Send to miners
- Score responses

**No restart needed!** Validator checks cache every cycle.

---

## 📊 How the Cache Updater Works

### Configuration (from `natix/validator/config.py`):

```python
IMAGE_DATASETS = {
    "Roadwork": [
        {"path": "natix-network-org/roadwork"},  # Hugging Face dataset
    ],
}

IMAGE_CACHE_UPDATE_INTERVAL = 1  # Update every 1 hour
IMAGE_PARQUET_CACHE_UPDATE_INTERVAL = 2  # Download new parquets every 2 hours
```

### What Gets Downloaded:

- **Dataset:** `natix-network-org/roadwork` on Hugging Face
- **Format:** Parquet files (compressed image data)
- **Images per parquet:** 100 images extracted
- **Total parquets:** 5 per dataset
- **Total images:** ~500 images
- **Update frequency:** Every 1-2 hours

### Cache Directory Structure:

```
~/.cache/natix/
├── Roadwork/
│   └── images/
│       ├── compressed/          # Parquet files downloaded from HF
│       │   ├── dataset_001.parquet
│       │   ├── dataset_002.parquet
│       │   └── ...
│       └── extracted/           # Images extracted from parquets
│           ├── image_001.jpg
│           ├── image_002.jpg
│           └── ... (up to 500 images)
└── Synthetic/
    ├── t2i/                     # Text-to-image generated (if you run generator)
    └── i2i/                     # Image-to-image generated (if you run generator)
```

---

## 🎯 STEP-BY-STEP: Complete Setup

### Step 1: Quick Test with Sample Image (Do this first!)

```bash
# Create directory
mkdir -p ~/.cache/natix/Roadwork/images/

# Copy sample image
cp /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet/neurons/unit_tests/sample_image.jpg \
   ~/.cache/natix/Roadwork/images/image_001.jpg

# Restart validator
pkill -f "neurons/validator.py"
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &

# Watch for queries!
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log | grep -E "(Sampling|challenge|query|miner)"
```

**Expected output:**
```
INFO | Sampling real image from real cache
INFO | Sampled image: image_001.jpg
INFO | Querying 1 miners...
INFO | Received response from UID 88
INFO | Miner 88 score: 0.85
```

### Step 2: Download Real Dataset (Do this next!)

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Start cache updater
./start_cache_updater.sh > /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log 2>&1 &

# Watch it work
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log
```

**Wait for:**
```
INFO | Starting image cache updater
INFO | Downloading parquet files from natix-network-org/roadwork
INFO | Extracting images from parquet files
INFO | Extracted 100 images to cache
```

### Step 3: Verify Everything Works

```bash
# Check how many images you have
ls -lh ~/.cache/natix/Roadwork/images/ | wc -l

# Watch validator use them
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log

# Watch miner respond
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log
```

---

## 🔧 Troubleshooting

### If Cache Updater Fails:

**Error: "Failed to download from Hugging Face"**
```bash
# Check internet connection
ping huggingface.co

# Try manual download test
poetry run python -c "from datasets import load_dataset; ds = load_dataset('natix-network-org/roadwork', split='train', streaming=True); print(next(iter(ds)))"
```

**Error: "No space left on device"**
```bash
# Check disk space
df -h ~/.cache/

# Clear old cache if needed
rm -rf ~/.cache/natix/Roadwork/images/compressed/*
```

### If Validator Still Shows "No images available":

```bash
# Check cache directory exists and has images
ls -la ~/.cache/natix/Roadwork/images/

# Check permissions
chmod -R 755 ~/.cache/natix/

# Restart validator
pkill -f "neurons/validator.py"
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &
```

---

## 📈 What to Expect

### Timeline:

1. **Minute 0:** Copy sample image → Validator can query immediately
2. **Minute 1:** Start cache updater → Begins downloading
3. **Minute 5-10:** First parquet downloaded → 100 images extracted
4. **Minute 10-20:** All parquets downloaded → ~500 images available
5. **Hour 1+:** Cache updater refreshes images automatically

### Validator Behavior:

**Before images:**
```
WARNING | No images available in cache
WARNING | Waiting for cache to populate. Challenge skipped.
```

**After images:**
```
INFO | Sampling real image from real cache
INFO | Sampled image: image_042.jpg
INFO | Querying 1 miners (UID: 88)
INFO | Received response from UID 88: {'label': 'roadwork', 'confidence': 0.87}
INFO | Miner 88 score: 0.87
```

### Miner Behavior:

**Before queries:**
```
INFO | Miner | UID:88 | Stake:0.000 | Trust:0.000 | Incentive:0.000 | Emission:0.000
(repeating every 5 seconds, no queries)
```

**After queries:**
```
INFO | Received image query from validator UID 89
INFO | Processing image with ViT detector
INFO | Prediction: roadwork (confidence: 0.87)
INFO | Sending response to validator
INFO | Miner | UID:88 | Stake:0.000 | Trust:0.000 | Incentive:0.000 | Emission:0.000
```

---

## 🎓 Understanding the Full System

### Production Validator Setup (3 Processes):

```
┌─────────────────────────────────────────────────────────┐
│                   VALIDATOR SYSTEM                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Process 1: Main Validator (neurons/validator.py)       │
│  ├─ Queries miners                                       │
│  ├─ Scores responses                                     │
│  ├─ Sets weights                                         │
│  └─ Status: ✅ Running (UID 89)                          │
│                                                          │
│  Process 2: Cache Updater (run_cache_updater.py)        │
│  ├─ Downloads images from Hugging Face                   │
│  ├─ Extracts to ~/.cache/natix/Roadwork/images/         │
│  ├─ Updates every 1-2 hours                              │
│  └─ Status: ⏳ We'll start this now!                     │
│                                                          │
│  Process 3: Data Generator (run_data_generator.py)      │
│  ├─ Generates synthetic images with AI                   │
│  ├─ Uses Stable Diffusion models                         │
│  ├─ Requires Hugging Face token                          │
│  └─ Status: ❌ Optional (not needed for testing)         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### For Testing, You Need:

- ✅ **Process 1 (Main Validator):** Already running!
- ✅ **Process 2 (Cache Updater):** We'll start this now!
- ❌ **Process 3 (Data Generator):** Optional, skip for testing

---

## 🎯 RECOMMENDED ACTION NOW

### Quick Win (2 minutes):

```bash
# Copy sample image
mkdir -p ~/.cache/natix/Roadwork/images/
cp /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet/neurons/unit_tests/sample_image.jpg \
   ~/.cache/natix/Roadwork/images/image_001.jpg

# Restart validator
pkill -f "neurons/validator.py"
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &

# Watch it work!
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log
```

### Full Setup (10 minutes):

```bash
# Start cache updater to download real dataset
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_cache_updater.sh > /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log 2>&1 &

# Monitor progress
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log
```

---

## ✅ Success Criteria

You'll know it's working when you see:

1. **Cache has images:**
   ```bash
   ls -lh ~/.cache/natix/Roadwork/images/ | wc -l
   # Should show > 0 files
   ```

2. **Validator logs show queries:**
   ```
   INFO | Sampling real image from real cache
   INFO | Querying miners...
   ```

3. **Miner logs show responses:**
   ```
   INFO | Received image query from validator
   INFO | Prediction: roadwork (confidence: 0.XX)
   ```

4. **No more "cache empty" warnings:**
   ```
   ❌ WARNING | No images available in cache  (OLD)
   ✅ INFO | Sampled image: image_042.jpg     (NEW)
   ```

---

**Ready to proceed?** Let me know and I'll run these commands for you!

**Summary:**
- ✅ NATIX provides free dataset on Hugging Face
- ✅ Cache updater script already exists
- ✅ Sample image available for immediate testing
- ✅ Everything is configured, just needs to be started!


