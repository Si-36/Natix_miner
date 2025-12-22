#!/bin/bash
# Start both miner and validator for testing

echo "🚀 Starting Testnet Miner and Validator"
echo "========================================"
echo ""

cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Check if miner is already running
if pgrep -f "neurons/miner.py" > /dev/null; then
    echo "✅ Miner is already running"
else
    echo "🔄 Starting miner in background..."
    nohup ./start_miner.sh > ../logs/miner.log 2>&1 &
    MINER_PID=$!
    echo "✅ Miner started (PID: $MINER_PID)"
    echo "   Logs: ../logs/miner.log"
    sleep 5
fi

# Check if validator is already running
if pgrep -f "neurons/validator.py" > /dev/null; then
    echo "✅ Validator is already running"
else
    echo "🔄 Starting validator in background..."
    nohup ./start_validator_testnet.sh > ../logs/validator.log 2>&1 &
    VALIDATOR_PID=$!
    echo "✅ Validator started (PID: $VALIDATOR_PID)"
    echo "   Logs: ../logs/validator.log"
    sleep 5
fi

echo ""
echo "📊 Process Status:"
echo "=================="
ps aux | grep -E "neurons/(miner|validator).py" | grep -v grep | awk '{printf "%-10s %-8s %s\n", $11, $2, $12" "$13" "$14}'

echo ""
echo "📋 Monitoring Commands:"
echo "======================="
echo ""
echo "Watch miner logs:"
echo "  tail -f /home/sina/projects/miner_b/phase0_testnet/logs/miner.log"
echo ""
echo "Watch validator logs:"
echo "  tail -f /home/sina/projects/miner_b/phase0_testnet/logs/validator.log"
echo ""
echo "Check metagraph:"
echo "  cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet"
echo "  poetry run btcli subnet show --netuid 323 --network test | grep -E '(88|89)'"
echo ""
echo "Stop both processes:"
echo "  pkill -f 'neurons/miner.py'"
echo "  pkill -f 'neurons/validator.py'"
echo ""
echo "✅ Setup complete! Monitor logs to see validator querying miner."

