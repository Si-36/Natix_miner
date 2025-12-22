#!/bin/bash

echo "🎯 NATIX Validator Image Cache Setup"
echo "===================================="
echo ""

# Step 1: Quick test with sample image
echo "📸 Step 1: Setting up quick test with sample image..."
mkdir -p ~/.cache/natix/Roadwork/images/

if [ -f "/home/sina/projects/miner_b/phase0_testnet/streetvision-subnet/neurons/unit_tests/sample_image.jpg" ]; then
    cp /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet/neurons/unit_tests/sample_image.jpg \
       ~/.cache/natix/Roadwork/images/image_001.jpg
    echo "✅ Sample image copied to cache"
else
    echo "⚠️  Sample image not found, skipping..."
fi

# Check current images
IMAGE_COUNT=$(ls ~/.cache/natix/Roadwork/images/*.jpg 2>/dev/null | wc -l)
echo "📊 Current images in cache: $IMAGE_COUNT"
echo ""

# Step 2: Restart validator
echo "🔄 Step 2: Restarting validator to pick up images..."
pkill -f "neurons/validator.py"
sleep 2

cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_validator_testnet.sh > /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>&1 &
VALIDATOR_PID=$!

echo "✅ Validator restarted (PID: $VALIDATOR_PID)"
echo ""

# Step 3: Start cache updater for real dataset
echo "📥 Step 3: Starting cache updater to download real dataset..."
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_cache_updater.sh > /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log 2>&1 &
CACHE_PID=$!

echo "✅ Cache updater started (PID: $CACHE_PID)"
echo ""

# Summary
echo "======================================"
echo "✅ SETUP COMPLETE!"
echo "======================================"
echo ""
echo "📊 Status:"
echo "  - Validator: Running (PID: $VALIDATOR_PID)"
echo "  - Cache Updater: Running (PID: $CACHE_PID)"
echo "  - Images in cache: $IMAGE_COUNT"
echo ""
echo "📁 Logs:"
echo "  - Validator: /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
echo "  - Cache Updater: /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log"
echo ""
echo "👀 Monitor progress:"
echo "  - Watch validator: tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
echo "  - Watch cache updater: tail -f /home/sina/projects/miner_b/phase0_testnet/logs/cache_updater.log"
echo "  - Check image count: watch -n 5 'ls ~/.cache/natix/Roadwork/images/*.jpg 2>/dev/null | wc -l'"
echo ""
echo "⏱️  Expected timeline:"
echo "  - Immediate: Validator uses sample image"
echo "  - 5-10 min: First batch of real images downloaded"
echo "  - 10-20 min: Full dataset (~500 images) available"
echo ""
echo "🎯 Success indicators:"
echo "  - Validator logs show: 'Sampling real image from real cache'"
echo "  - Validator logs show: 'Querying miners...'"
echo "  - Miner logs show: 'Received image query from validator'"
echo "  - No more: 'No images available in cache' warnings"
echo ""


