#!/bin/bash
# Run local miner test using Poetry environment

cd "$(dirname "$0")/streetvision-subnet" || exit 1

echo "🧪 Running local miner test with Poetry..."
echo ""

poetry run python ../test_miner_local.py



