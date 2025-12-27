#!/bin/bash
# Download ROADWork Dataset via Hugging Face
# Dataset: hayden-yuma/roadwork
# Size: ~8,549 images (6,250 train, 2,300 test)

set -e  # Exit on error

echo "================================================================================"
echo "DOWNLOADING ROADWORK DATASET VIA HUGGING FACE"
echo "================================================================================"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing Hugging Face CLI..."
    pip install -U "huggingface_hub[cli]"
fi

# Create download directory
mkdir -p ~/Natix_miner/data/roadwork_hf

echo ""
echo "📥 Downloading ROADWork dataset (hayden-yuma/roadwork)..."
echo "   This is a PUBLIC dataset - no login required"
echo ""

# Download dataset (no authentication needed)
huggingface-cli download \
    --repo-type dataset \
    hayden-yuma/roadwork \
    --local-dir ~/Natix_miner/data/roadwork_hf

echo ""
echo "✅ ROADWork dataset downloaded successfully!"
echo "📁 Location: ~/Natix_miner/data/roadwork_hf"
echo ""

# Show directory structure
echo "📊 Dataset structure:"
du -sh ~/Natix_miner/data/roadwork_hf
ls -lh ~/Natix_miner/data/roadwork_hf/ | head -20

# Check for parquet files (HuggingFace format)
echo ""
echo "📋 Dataset files:"
find ~/Natix_miner/data/roadwork_hf -name "*.parquet" | head -5

echo ""
echo "================================================================================"
echo "NEXT STEP: Run filter_datasets_smart.py to prepare training data"
echo "================================================================================"
