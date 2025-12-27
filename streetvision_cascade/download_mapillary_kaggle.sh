#!/bin/bash
# Download Mapillary Vistas via Kaggle API
# Dataset: kaggleprollc/mapillary-vistas-image-data-collection
# Size: ~21GB

set -e  # Exit on error

echo "================================================================================"
echo "DOWNLOADING MAPILLARY VISTAS DATASET VIA KAGGLE"
echo "================================================================================"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle API..."
    pip install kaggle
fi

# Check for kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "⚠️  Kaggle API credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Create download directory
mkdir -p ~/Natix_miner/data/mapillary_raw
mkdir -p ~/Natix_miner/data/mapillary_vistas

echo ""
echo "📥 Downloading Mapillary Vistas (~21GB, may take 30-60 minutes)..."
echo ""

# Download dataset
kaggle datasets download -d kaggleprollc/mapillary-vistas-image-data-collection \
    -p ~/Natix_miner/data/mapillary_raw

echo ""
echo "📂 Extracting dataset..."
echo ""

# Unzip
cd ~/Natix_miner/data
unzip -q mapillary_raw/mapillary-vistas-image-data-collection.zip -d mapillary_vistas

echo ""
echo "✅ Mapillary Vistas downloaded successfully!"
echo "📁 Location: ~/Natix_miner/data/mapillary_vistas"
echo ""

# Show directory structure
echo "📊 Dataset structure:"
du -sh mapillary_vistas
ls -lh mapillary_vistas/ | head -20

echo ""
echo "================================================================================"
echo "NEXT STEP: Run download_roadwork_hf.sh to get ROADWork dataset"
echo "================================================================================"
