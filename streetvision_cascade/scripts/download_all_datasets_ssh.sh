#!/bin/bash
# Complete Dataset Download Script for SSH
# Downloads all datasets directly on SSH server

set -e  # Exit on error

echo "======================================================================"
echo "🚀 StreetVision Complete Dataset Download (SSH)"
echo "======================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if in correct directory
if [ ! -f "train_stage1_head.py" ]; then
    echo -e "${RED}❌ Error: Must run from streetvision_cascade directory${NC}"
    echo "   cd ~/Natix_miner/streetvision_cascade"
    exit 1
fi

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Activated virtual environment${NC}"
fi

# Create data directories
mkdir -p data/{natix_official,roadwork_iccv,roadwork_extra,open_images,gtsrb_class25,kaggle_road_issues}

echo ""
echo "======================================================================"
echo "📥 Step 1: NATIX Official Dataset"
echo "======================================================================"

if [ -f "data/natix_official/train_labels.csv" ]; then
    echo -e "${YELLOW}⚠️  NATIX already exists, skipping...${NC}"
    echo "   To re-download, delete data/natix_official/ first"
else
    echo "Downloading NATIX from Hugging Face..."
    python3 << 'PY'
from datasets import load_dataset
from pathlib import Path
import os

print("📥 Downloading NATIX official dataset...")
print("   This may take 10-20 minutes (~8GB)")

try:
    dataset = load_dataset("natix-network-org/roadwork", split="train")
    output_dir = Path("data/natix_official/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    dataset.to_parquet(str(output_dir / "train.parquet"), batch_size=1000)
    
    print(f"✅ NATIX downloaded: {len(dataset)} samples")
    
    # Convert to images + CSV
    print("Converting parquet to images + CSV...")
    os.system("python3 convert_natix_parquet.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Make sure you have access to natix-network-org/roadwork")
    print("   You may need to accept terms on Hugging Face")
PY
fi

echo ""
echo "======================================================================"
echo "📥 Step 2: ROADWork ICCV 2025 Dataset"
echo "======================================================================"

if [ -f "data/roadwork_iccv/train_labels.csv" ]; then
    echo -e "${YELLOW}⚠️  ROADWork already processed, skipping...${NC}"
    echo "   To re-process, delete data/roadwork_iccv/train_labels.csv first"
else
    echo -e "${YELLOW}⚠️  ROADWork requires manual download${NC}"
    echo ""
    echo "   Steps:"
    echo "   1. Visit: https://github.com/anuragxel/roadwork-dataset"
    echo "   2. Follow their download instructions (CMU KiltHub or Google Drive)"
    echo "   3. Download images.zip and annotations.zip"
    echo "   4. Extract to: data/roadwork_iccv/raw/"
    echo "   5. Then run: python3 prepare_roadwork_data.py --process_roadwork"
    echo ""
    read -p "Press Enter after downloading ROADWork raw data, or 's' to skip: " choice
    if [ "$choice" != "s" ]; then
        if [ -d "data/roadwork_iccv/raw/annotations" ] && [ -d "data/roadwork_iccv/raw/images" ]; then
            echo "Processing ROADWork..."
            python3 prepare_roadwork_data.py --process_roadwork
        else
            echo -e "${RED}❌ ROADWork raw data not found in data/roadwork_iccv/raw/${NC}"
        fi
    fi
fi

echo ""
echo "======================================================================"
echo "📥 Step 3: Roboflow Roadwork Extras"
echo "======================================================================"

if [ -f "data/roadwork_extra/train_labels.csv" ]; then
    echo -e "${YELLOW}⚠️  Roboflow already processed, skipping...${NC}"
else
    echo -e "${YELLOW}⚠️  Roboflow requires manual download${NC}"
    echo ""
    echo "   Steps:"
    echo "   1. Visit: https://universe.roboflow.com/workzone/roadwork"
    echo "   2. Click 'Download Dataset'"
    echo "   3. Choose COCO JSON format"
    echo "   4. Extract ZIP to: data/roadwork_extra/raw/"
    echo "   5. Then run: python3 prepare_roadwork_data.py --process_extra"
    echo ""
    read -p "Press Enter after downloading Roboflow data, or 's' to skip: " choice
    if [ "$choice" != "s" ]; then
        if [ -d "data/roadwork_extra/raw" ] && [ "$(ls -A data/roadwork_extra/raw/*.jpg 2>/dev/null)" ]; then
            echo "Processing Roboflow..."
            python3 prepare_roadwork_data.py --process_extra
        else
            echo -e "${RED}❌ Roboflow images not found in data/roadwork_extra/raw/${NC}"
        fi
    fi
fi

echo ""
echo "======================================================================"
echo "📥 Step 4: Open Images V7 (Positives Only)"
echo "======================================================================"

if [ -f "data/open_images/train_labels.csv" ]; then
    echo -e "${YELLOW}⚠️  Open Images already exists, skipping...${NC}"
else
    echo "Downloading Open Images positives..."
    python3 download_open_images_positives_only_no_mongo.py
fi

echo ""
echo "======================================================================"
echo "📥 Step 5: GTSRB Class 25 (EU Roadwork Signs)"
echo "======================================================================"

if [ -f "data/gtsrb_class25/train_labels.csv" ]; then
    echo -e "${YELLOW}⚠️  GTSRB already exists, skipping...${NC}"
else
    echo "Downloading GTSRB..."
    mkdir -p data/gtsrb_class25/raw
    cd data/gtsrb_class25/raw
    
    if [ ! -f "GTSRB_Final_Training_Images.zip" ]; then
        echo "Downloading GTSRB training images..."
        wget -q --show-progress https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
    fi
    
    if [ ! -d "GTSRB/Final_Training/Images" ]; then
        echo "Extracting GTSRB..."
        unzip -q GTSRB_Final_Training_Images.zip
    fi
    
    cd ../../..
    
    echo "Converting GTSRB Class 25..."
    python3 convert_gtsrb_class25.py
fi

echo ""
echo "======================================================================"
echo "📥 Step 6: Kaggle Road Issues (Negatives)"
echo "======================================================================"

if [ -f "data/kaggle_road_issues/train_labels.csv" ]; then
    echo -e "${YELLOW}⚠️  Kaggle Road Issues already exists, skipping...${NC}"
else
    echo -e "${YELLOW}⚠️  Kaggle requires manual download${NC}"
    echo ""
    echo "   Steps:"
    echo "   1. Visit: https://www.kaggle.com/datasets (search 'road issues')"
    echo "   2. Download dataset ZIP"
    echo "   3. Extract to: data/kaggle_road_issues/raw/"
    echo "   4. Then run: python3 convert_kaggle_road_issues.py"
    echo ""
    read -p "Press Enter after downloading Kaggle data, or 's' to skip: " choice
    if [ "$choice" != "s" ]; then
        if [ -d "data/kaggle_road_issues/raw" ] && [ "$(ls -A data/kaggle_road_issues/raw/*.jpg 2>/dev/null)" ]; then
            echo "Converting Kaggle Road Issues..."
            python3 convert_kaggle_road_issues.py
        else
            echo -e "${RED}❌ Kaggle images not found in data/kaggle_road_issues/raw/${NC}"
        fi
    fi
fi

echo ""
echo "======================================================================"
echo "✅ Step 7: Verify All Datasets"
echo "======================================================================"

python3 verify_datasets.py --check_all

echo ""
echo "======================================================================"
echo "🎉 Dataset Download Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Train baseline: python3 train_stage1_head.py --mode train --epochs 10"
echo "  2. Train aggressive: python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork"
echo "  3. Validate thresholds: python3 validate_thresholds.py"
echo "  4. Test cascade: python3 test_cascade_small.py"
echo ""

