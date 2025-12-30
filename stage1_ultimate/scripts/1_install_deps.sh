#!/bin/bash
# Script 1: Install Python Dependencies
# Run after cloning repo
# Usage: cd ~/natix/stage1_ultimate && bash scripts/1_install_deps.sh

set -e
echo "==================================================================="
echo "Installing Python Dependencies"
echo "==================================================================="
echo ""

# Check Python version
echo "Python version:"
python3 --version
echo ""

# Create venv
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Install training dependencies
echo "Installing training dependencies..."
pip install lightning transformers hydra-core omegaconf pillow numpy pandas scikit-learn matplotlib seaborn tqdm
echo ""

# Install HuggingFace datasets
echo "Installing datasets library..."
pip install datasets huggingface_hub
echo ""

# Verify installation
echo "==================================================================="
echo "Verifying Installation"
echo "==================================================================="
python << 'EOF'
import torch
import lightning
import transformers
import datasets

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"✅ Lightning: {lightning.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ Datasets: {datasets.__version__}")
EOF

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next: bash scripts/2_download_data.sh"
