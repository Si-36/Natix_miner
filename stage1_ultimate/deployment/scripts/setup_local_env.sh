#!/bin/bash
# Day 1 Setup Script - Creates complete local development environment

set -e

PROJECT_ROOT="/home/sina/projects/miner_b/stage1_ultimate"
cd "$PROJECT_ROOT"

echo "🚀 Setting up local development environment..."
echo "================================================"

# 1. Check Python version
echo ""
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc) -eq 1 ]]; then
    echo "✅ Python $PYTHON_VERSION (compatible)"
else
    echo "❌ Python $PYTHON_VERSION (requires 3.10+)"
    exit 1
fi

# 2. Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
if [ ! -d ".venv_2026" ]; then
    python3 -m venv .venv_2026
    echo "✅ Virtual environment created at .venv_2026"
else
    echo "ℹ️  Virtual environment already exists"
fi

# 3. Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
source .venv_2026/bin/activate
echo "✅ Virtual environment activated"

# 4. Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ Pip upgraded"

# 5. Install local testing dependencies
echo ""
echo "Step 5: Installing local testing dependencies..."
pip install -r deployment/requirements_local_test.txt
echo "✅ Local dependencies installed"

# 6. Verify installations
echo ""
echo "Step 6: Verifying installations..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} (CPU)')"
python3 -c "import pytest; print(f'✅ pytest {pytest.__version__}')"
python3 -c "import transformers; print(f'✅ transformers {transformers.__version__}')"

# 7. Create .env file
echo ""
echo "Step 7: Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Local Development Environment
ENVIRONMENT=local
DEBUG=true
LOG_LEVEL=DEBUG

# Paths
PROJECT_ROOT=/home/sina/projects/miner_b/stage1_ultimate
CACHE_DIR=/home/sina/.cache/huggingface

# HuggingFace (optional for local testing)
HF_TOKEN=

# GPU Settings (for production)
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.95
EOF
    echo "✅ .env file created"
else
    echo "ℹ️  .env file already exists"
fi

# 8. Run initial tests
echo ""
echo "Step 8: Running initial test..."
python3 -c "print('✅ Python environment working correctly')"

echo ""
echo "================================================"
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv_2026/bin/activate"
echo "2. Start Day 2: Create mock infrastructure"
echo "3. Run tests: pytest tests/ -v"
