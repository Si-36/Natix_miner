#!/bin/bash

echo "🚀 Restarting NATIX Testnet Setup After Reboot"
echo "=============================================="
echo ""

cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Kill any existing processes (just in case)
pkill -f "neurons/miner.py" 2>/dev/null
pkill -f "neurons/validator.py" 2>/dev/null
pkill -f "run_cache_updater.py" 2>/dev/null
sleep 3

echo "✅ Cleaned up any existing processes"
echo ""

# Check if images are still in cache
IMAGE_COUNT=$(ls ~/.cache/natix/Roadwork/image/*.jpeg 2>/dev/null | wc -l)
echo "📁 Images in cache: $IMAGE_COUNT"
echo ""

# Start miner
echo "🔧 Starting Miner (UID 88)..."
./start_miner.sh > /home/sina/projects/miner_b/phase0_testnet/logs/miner.log 2>&1 &
MINER_PID=$!
echo "   Miner PID: $MINER_PID"
sleep 5

# Start validator
echo "🔧 Starting Validator (UID 89)..."
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &
VALIDATOR_PID=$!
echo "   Validator PID: $VALIDATOR_PID"
sleep 5

# Start cache updater
echo "🔧 Starting Cache Updater..."
./start_cache_updater.sh > /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log 2>&1 &
CACHE_PID=$!
echo "   Cache Updater PID: $CACHE_PID"
sleep 3

echo ""
echo "=============================================="
echo "✅ ALL PROCESSES STARTED!"
echo "=============================================="
echo ""

# Show status
echo "📊 Process Status:"
ps aux | grep -E "(neurons/miner|neurons/validator|run_cache_updater)" | grep python | grep -v grep | awk '{print "   "$12" "$13" "$14" (PID: "$2")"}'
echo ""

echo "📁 Files:"
echo "   Images: $IMAGE_COUNT"
echo "   Logs: /home/sina/projects/miner_b/phase0_testnet/logs/"
echo ""

echo "👀 Monitor Progress:"
echo "   Miner:     tail -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log"
echo "   Validator: tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
echo ""

echo "⏱️  Give it 60-90 seconds for validator to initialize, then it should start querying the miner!"
echo ""

