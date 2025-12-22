#!/bin/bash

echo "🚀 STARTING LOCAL MINER-VALIDATOR TESTING"
echo "=========================================="
echo ""

cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "neurons/miner.py" 2>/dev/null
pkill -f "neurons/validator.py" 2>/dev/null
sleep 3
echo "✅ Cleaned up"
echo ""

# Check images
IMAGE_COUNT=$(ls ~/.cache/natix/Roadwork/image/*.jpeg 2>/dev/null | wc -l)
echo "📁 Images in cache: $IMAGE_COUNT"
if [ "$IMAGE_COUNT" -lt 10 ]; then
    echo "⚠️  WARNING: Less than 10 images! Validator may not work properly."
    echo "   Run ./quick_setup_images.sh to download more images."
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Create logs directory
mkdir -p /home/sina/projects/miner_b/phase0_testnet/logs

# Start Miner (with localhost configuration)
echo "🔧 Starting Miner (UID 88) on localhost..."
./start_miner_local.sh > /home/sina/projects/miner_b/phase0_testnet/logs/miner.log 2>&1 &
MINER_PID=$!
echo "   Miner PID: $MINER_PID"
echo "   Waiting for miner to initialize..."
sleep 10

# Check if miner is running
if ps -p $MINER_PID > /dev/null; then
    echo "   ✅ Miner is running"
else
    echo "   ❌ Miner failed to start!"
    echo "   Check logs: tail -50 /home/sina/projects/miner_b/phase0_testnet/logs/miner.log"
    exit 1
fi
echo ""

# Check if miner is listening
echo "🔍 Checking if miner is listening on port 8091..."
sleep 2
if ss -tuln | grep :8091 > /dev/null; then
    echo "   ✅ Miner is listening on port 8091"
else
    echo "   ⚠️  Miner may not be listening yet (still starting up)"
fi
echo ""

# Start Validator
echo "🔧 Starting Validator (UID 89)..."
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &
VALIDATOR_PID=$!
echo "   Validator PID: $VALIDATOR_PID"
echo "   Waiting for validator to initialize..."
sleep 10

# Check if validator is running
if ps -p $VALIDATOR_PID > /dev/null; then
    echo "   ✅ Validator is running"
else
    echo "   ❌ Validator failed to start!"
    echo "   Check logs: tail -50 /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ BOTH PROCESSES STARTED SUCCESSFULLY!"
echo "=========================================="
echo ""

# Show process status
echo "📊 Process Status:"
ps aux | grep -E "(neurons/miner|neurons/validator)" | grep python | grep -v grep | awk '{print "   "$12" "$13" "$14" (PID: "$2")"}'
echo ""

# Show listening ports
echo "🔌 Listening Ports:"
ss -tuln | grep -E "8091|8092" | awk '{print "   "$5}' || echo "   (Ports may still be binding...)"
echo ""

echo "📁 Files:"
echo "   Images: $IMAGE_COUNT"
echo "   Logs: /home/sina/projects/miner_b/phase0_testnet/logs/"
echo ""

echo "👀 Monitor in Real-Time:"
echo "   Miner:     tail -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log"
echo "   Validator: tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
echo ""
echo "   Or watch both:"
echo "   tail -f /home/sina/projects/miner_b/phase0_testnet/logs/*.log"
echo ""

echo "⏱️  Timeline:"
echo "   • Validator will initialize in ~60-90 seconds"
echo "   • Then it will start querying miners every ~12 seconds"
echo "   • Look for 'Sending image challenge' in validator.log"
echo "   • Look for UID 88 (your miner) in the miner list"
echo "   • Check if miner receives and responds to queries"
echo ""

echo "🎯 What to Look For:"
echo ""
echo "   In VALIDATOR LOG:"
echo "   ✅ 'Sampling real image from real cache'"
echo "   ✅ 'Miner UIDs: [... 88 ...]' (your miner selected)"
echo "   ✅ 'Sending image challenge to X miners'"
echo ""
echo "   In MINER LOG:"
echo "   ✅ 'Received forward request' or 'Processing image'"
echo "   ✅ 'Prediction: 0.XXXX' (miner responding)"
echo "   ✅ 'Sending response to validator'"
echo ""

echo "🚨 If Miner Doesn't Respond:"
echo "   • Check miner.log for errors"
echo "   • Verify port 8091 is accessible: curl http://localhost:8091"
echo "   • Check miner axon IP in metagraph (should be 127.0.0.1)"
echo ""

echo "=========================================="
echo "🎬 Monitoring validator logs now..."
echo "   (Press Ctrl+C to stop monitoring and return to terminal)"
echo "=========================================="
echo ""

sleep 5

# Follow validator logs
tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log
