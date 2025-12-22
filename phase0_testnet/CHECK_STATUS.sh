#!/bin/bash

echo "📊 MINER-VALIDATOR STATUS CHECK"
echo "================================"
echo ""

# Check processes
echo "🔍 Process Status:"
MINER_PID=$(ps aux | grep "neurons/miner.py" | grep -v grep | awk '{print $2}')
VALIDATOR_PID=$(ps aux | grep "neurons/validator.py" | grep -v grep | awk '{print $2}')

if [ -n "$MINER_PID" ]; then
    echo "   ✅ Miner running (PID: $MINER_PID)"
else
    echo "   ❌ Miner NOT running"
fi

if [ -n "$VALIDATOR_PID" ]; then
    echo "   ✅ Validator running (PID: $VALIDATOR_PID)"
else
    echo "   ❌ Validator NOT running"
fi
echo ""

# Check ports
echo "🔌 Port Status:"
if ss -tuln | grep :8091 > /dev/null; then
    echo "   ✅ Port 8091 (Miner) is listening"
else
    echo "   ❌ Port 8091 (Miner) is NOT listening"
fi

if ss -tuln | grep :8092 > /dev/null; then
    echo "   ✅ Port 8092 (Validator) is listening"
else
    echo "   ⚠️  Port 8092 (Validator) is NOT listening"
fi
echo ""

# Check images
IMAGE_COUNT=$(ls ~/.cache/natix/Roadwork/image/*.jpeg 2>/dev/null | wc -l)
echo "📁 Cache Status:"
echo "   Images: $IMAGE_COUNT"
echo ""

# Check recent logs
echo "📝 Recent Activity (last 5 lines):"
echo ""
echo "   MINER LOG:"
if [ -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log ]; then
    tail -5 /home/sina/projects/miner_b/phase0_testnet/logs/miner.log | sed 's/^/      /'
else
    echo "      No log file found"
fi
echo ""

echo "   VALIDATOR LOG:"
if [ -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log ]; then
    tail -5 /home/sina/projects/miner_b/phase0_testnet/logs/validator.log | sed 's/^/      /'
else
    echo "      No log file found"
fi
echo ""

# Check for queries/responses
echo "🔎 Query/Response Check:"
if [ -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log ]; then
    QUERY_COUNT=$(grep -c "Sending image challenge" /home/sina/projects/miner_b/phase0_testnet/logs/validator.log 2>/dev/null || echo "0")
    echo "   Queries sent by validator: $QUERY_COUNT"
fi

if [ -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log ]; then
    RECEIVED_COUNT=$(grep -c -E "forward|Received|Processing" /home/sina/projects/miner_b/phase0_testnet/logs/miner.log 2>/dev/null || echo "0")
    echo "   Requests received by miner: $RECEIVED_COUNT"
fi
echo ""

echo "================================"
echo "💡 Quick Actions:"
echo "   View miner logs:     tail -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log"
echo "   View validator logs: tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
echo "   Stop all:            pkill -f 'neurons/(miner|validator).py'"
echo "   Restart:             ./START_LOCAL_TESTING.sh"
echo ""
