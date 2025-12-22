# Phase 0 Testnet Setup Status

## ✅ Completed Tasks

1. **Environment Setup**
   - ✅ Python 3.11.14 installed
   - ✅ Poetry installed and configured
   - ✅ Virtual environment created with Poetry (Python 3.11)
   - ✅ All NATIX dependencies installed (133 packages)
   - ✅ CUDA verified: PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3070 Laptop GPU detected

2. **Repository Setup**
   - ✅ NATIX StreetVision subnet cloned
   - ✅ miner.env configuration file created with testnet settings

3. **Configuration**
   - ✅ NETUID set to 323 (testnet)
   - ✅ SUBTENSOR_NETWORK set to "test"
   - ✅ IMAGE_DETECTOR_DEVICE set to "cuda"
   - ✅ Model configured: ViT with ViT_roadwork.yaml
   - ✅ Model will auto-download from: `natix-network-org/roadwork` (Hugging Face)

## ⏳ Next Steps (Manual/Interactive)

### 1. Create Wallets (Interactive - Run these commands)

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
poetry run btcli wallet new_coldkey --wallet.name testnet_wallet
# When prompted, enter a password and save the mnemonic phrase securely!

poetry run btcli wallet new_hotkey --wallet.name testnet_wallet --wallet.hotkey miner_hotkey
# When prompted, enter the same password and save the hotkey mnemonic!
```

**IMPORTANT:** Save both mnemonics in a secure location! You'll need them to restore wallets.

### 2. Update miner.env with Wallet Names

After creating wallets, the miner.env already has:
- WALLET_NAME=testnet_wallet
- WALLET_HOTKEY=miner_hotkey

These match what we'll create above, so no changes needed.

### 3. Join Discord and Request Testnet TAO

1. Join Bittensor Discord: https://discord.gg/bittensor
2. Navigate to #testnet-faucet channel
3. Request testnet TAO with your wallet address:

```bash
# Get your wallet address first:
poetry run btcli wallet overview --wallet.name testnet_wallet --network test

# Then request in Discord:
# "Request testnet TAO for miner testing on netuid 323. Wallet: [YOUR_ADDRESS]"
```

### 4. Verify Testnet TAO Received

```bash
poetry run btcli wallet balance --wallet.name testnet_wallet --network test
# Should show ~100+ test TAO after Discord request is processed (usually 24-48 hours)
```

### 5. Register on Testnet

Once you have testnet TAO:

```bash
poetry run btcli subnet register \
  --netuid 323 \
  --wallet.name testnet_wallet \
  --wallet.hotkey miner_hotkey \
  --network test
```

### 6. Register with NATIX Application Server

After registration:

```bash
# Get your UID first:
poetry run btcli wallet overview --wallet.name testnet_wallet --network test

# Then register with NATIX (replace [YOUR_UID] with your actual UID):
./register.sh [YOUR_UID] testnet_wallet miner_hotkey miner natix-network-org/roadwork
```

### 7. Test Model Loading Locally (Before Testnet)

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
poetry run python -c "
from base_miner.detectors import DETECTOR_REGISTRY
import torch

# Check VRAM before
print(f'VRAM before: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

# Load model
detector = DETECTOR_REGISTRY['ViT'](
    config_name='ViT_roadwork.yaml',
    device='cuda'
)

# Check VRAM after
print(f'VRAM after: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print('✅ Model loaded successfully!')
"
```

### 8. Start Miner on Testnet

Once registered and model tested:

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
source miner.env  # Load environment variables
poetry run python run_neuron.py --neuron.name miner --netuid 323 --subtensor.network test
```

Or use the start script:

```bash
./start_miner.sh
```

## 📊 Current Configuration

**Hardware:**
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM)
- CUDA: 12.4
- PyTorch: 2.6.0+cu124

**Software:**
- Python: 3.11.14 (Poetry environment)
- Bittensor: 9.12.2 (via Poetry)
- NATIX StreetVision: Latest from GitHub

**Testnet Settings:**
- NETUID: 323
- Network: test
- Endpoint: wss://test.finney.opentensor.ai:443
- Axon Port: 8091

**Model:**
- Type: ViT (Vision Transformer)
- Config: ViT_roadwork.yaml
- Source: natix-network-org/roadwork (Hugging Face)
- Device: cuda

## 🎯 Success Criteria (Phase 0)

- [ ] Miner receives validator queries on testnet
- [ ] Predictions return 0.0-1.0 range (95%+ success rate)
- [ ] Accuracy: >85% on validator challenges
- [ ] Latency: <150ms average
- [ ] Stability: 24-hour continuous operation
- [ ] VRAM usage: <6GB (safe margin on 8GB card)

## 📝 Notes

- The model will download automatically from Hugging Face on first use (~343MB)
- Monitor VRAM usage with: `watch -n 1 nvidia-smi`
- Check miner logs in: `logs/` directory
- Testnet TAO requests typically take 24-48 hours to process

## 🚨 Troubleshooting

**If model download fails:**
- Check internet connection
- Try manually: `huggingface-cli download natix-network-org/roadwork`

**If VRAM overflow:**
- Set IMAGE_DETECTOR_DEVICE=cpu temporarily to test
- Or use FP16 precision (will need code modification)

**If registration fails:**
- Verify you have testnet TAO balance
- Check subnet is accessible: `poetry run btcli subnet show --netuid 323 --network test`

