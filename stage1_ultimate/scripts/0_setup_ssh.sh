#!/bin/bash
# Script 0: SSH Environment Setup
# Run this FIRST on rental GPU machine
# Usage: bash scripts/0_setup_ssh.sh

set -e
echo "==================================================================="
echo "NATIX Stage-1: SSH Environment Setup"
echo "==================================================================="
echo ""

# Check GPUs
echo "1. Checking GPUs..."
nvidia-smi
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPU_COUNT GPU(s)"
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2+ GPUs!"
fi
echo ""

# Check disk space
echo "2. Checking disk space..."
df -h ~
echo ""

# Create directories
echo "3. Creating directory structure..."
mkdir -p ~/natix/data
mkdir -p ~/natix/runs
mkdir -p ~/natix/logs
ls -la ~/natix/
echo ""

# Install tmux
echo "4. Installing tmux (for persistent sessions)..."
if ! command -v tmux &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y tmux
fi
echo "tmux version: $(tmux -V)"
echo ""

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Clone repo: cd ~/natix && git clone <repo> stage1_ultimate"
echo "  2. Run: bash scripts/1_install_deps.sh"
