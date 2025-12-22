# 🔬 Setting Up Your Own Validator for Testing

**Date:** December 19, 2025  
**Purpose:** Run your own validator to test your miner on testnet

---

## 📋 Overview

Since NATIX focuses on mainnet and testnet validators aren't active, you can run your own validator to test your miner. This allows you to:

1. ✅ Test your miner receives queries
2. ✅ Verify predictions are working
3. ✅ Test the full query → response pipeline
4. ✅ Validate your setup before mainnet

---

## 🎯 Step-by-Step Setup

### Step 1: Update Miner Registration (Dev Endpoint)

Your miner needs to register with the **dev** endpoint:

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Registration has been updated to use dev endpoint
poetry run python register_natix.py
```

**Changes made:**
- ✅ Updated `register_natix.py` to use `https://hydra.dev.natix.network/participant/register`
- ✅ Updated `miner.env` to set `PROXY_CLIENT_URL=https://hydra.dev.natix.network`

### Step 2: Create Validator Wallet/Hotkey

You need a separate hotkey for your validator (different from miner hotkey):

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Create validator hotkey in your testnet wallet
poetry run btcli wallet new_hotkey \
  --wallet.name testnet_wallet \
  --wallet.hotkey validator_hotkey \
  --subtensor.network test
```

### Step 3: Register Validator on Testnet

Register your validator on Bittensor testnet:

```bash
# Register validator on subnet 323 (testnet)
poetry run btcli subnet register \
  --netuid 323 \
  --wallet.name testnet_wallet \
  --wallet.hotkey validator_hotkey \
  --subtensor.network test
```

**Note:** You'll need testnet TAO for registration. If you don't have enough, request more from Bittensor Discord.

### Step 4: Register Validator with NATIX (Dev)

After getting your validator UID, register with NATIX dev server:

```bash
# Update register_natix.py with validator UID, then run:
poetry run python register_natix.py

# OR use the shell script:
./register.sh <VALIDATOR_UID> testnet_wallet validator_hotkey validator
```

**Important:** Change the `type` to `"validator"` in the registration payload!

### Step 5: Configure Validator Environment

The `validator.env` file has been created with testnet settings. Update it:

```bash
# Edit validator.env
nano validator.env
```

**Required changes:**
- Set your actual `WANDB_API_KEY` (or use a dummy value for testing)
- Set your `HUGGING_FACE_TOKEN` (required for synthetic image generation)
- Verify `WALLET_NAME=testnet_wallet`
- Verify `WALLET_HOTKEY=validator_hotkey`
- Verify `PROXY_CLIENT_URL=https://hydra.dev.natix.network`

### Step 6: Start Validator

**Option A: Direct Start (for testing)**

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Make sure miner is running in another terminal first
# Then start validator
source validator.env
poetry run python neurons/validator.py \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $VALIDATOR_AXON_PORT \
  --proxy.port $VALIDATOR_PROXY_PORT \
  --proxy.proxy_client_url $PROXY_CLIENT_URL \
  --logging.debug
```

**Option B: Use start_validator.sh (recommended)**

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./start_validator.sh
```

---

## 🔍 Testing Your Miner

Once validator is running, it will:

1. **Query miners every ~30 seconds** with image challenges
2. **Your miner (UID 88) should receive queries**
3. **Check validator logs** to see predictions from your miner
4. **Check miner logs** to see incoming queries

**Monitor logs:**

```bash
# Terminal 1: Miner logs
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
tail -f logs/miner.log  # Or wherever your miner logs are

# Terminal 2: Validator logs  
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
tail -f logs/validator.log  # Or wherever validator logs are
```

**What to look for:**
- ✅ Miner logs: "Received image challenge!"
- ✅ Miner logs: "PREDICTION = 0.xxxx"
- ✅ Validator logs: Responses from UID 88
- ✅ Validator logs: Predictions received from your miner

---

## ⚠️ Important Notes

### Resource Requirements

Running a validator requires:
- **GPU:** For synthetic image generation (same GPU can run both miner + validator if you have enough VRAM)
- **RAM:** Validator uses significant memory
- **Disk:** Cache directories for real/synthetic images

**If you only have one GPU:**
- You can run miner on CPU and validator on GPU (validator needs GPU more)
- OR run validator only (it will test your miner)

### Testnet TAO Requirements

- **Validator registration:** Requires testnet TAO (request from Bittensor Discord)
- **Operations:** Validator needs TAO for operations (very minimal on testnet)

### Synthetic Image Generation

Validator generates synthetic images using Hugging Face models. You need:
- ✅ Hugging Face account
- ✅ Hugging Face token (free)
- ✅ GPU for image generation (or use CPU, but slower)

---

## 🎯 Expected Behavior

### If Everything Works:

1. **Validator starts successfully**
   - Connects to testnet
   - Loads models (real images, synthetic generator)
   - Starts querying miners

2. **Miner receives queries**
   - Logs: "Received image challenge!"
   - Processes image
   - Returns prediction

3. **Validator receives responses**
   - Logs predictions from UID 88
   - Calculates rewards (on testnet, rewards are fake)
   - Continues querying

### If Miner Not Receiving Queries:

Check:
- ✅ Miner is running and connected
- ✅ Miner axon is serving (`axon.is_serving = True`)
- ✅ Validator can see miner in metagraph
- ✅ Check validator logs for errors

---

## 🚀 Quick Start Summary

```bash
# 1. Register miner with dev endpoint (already done)
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
poetry run python register_natix.py

# 2. Create validator hotkey
poetry run btcli wallet new_hotkey \
  --wallet.name testnet_wallet \
  --wallet.hotkey validator_hotkey \
  --subtensor.network test

# 3. Register validator
poetry run btcli subnet register \
  --netuid 323 \
  --wallet.name testnet_wallet \
  --wallet.hotkey validator_hotkey \
  --subtensor.network test

# 4. Register validator with NATIX
# (Update register_natix.py with validator UID and type="validator", then run)
poetry run python register_natix.py

# 5. Update validator.env with your tokens

# 6. Start miner (Terminal 1)
./start_miner.sh

# 7. Start validator (Terminal 2)
./start_validator.sh
```

---

## 📝 Files Updated

1. ✅ `register_natix.py` - Updated to use `https://hydra.dev.natix.network`
2. ✅ `miner.env` - Updated `PROXY_CLIENT_URL=https://hydra.dev.natix.network`
3. ✅ `validator.env` - Created with testnet configuration

---

## 🎓 What This Proves

Running your own validator allows you to:
- ✅ Validate miner receives and processes queries
- ✅ Verify predictions are in correct format
- ✅ Test full pipeline end-to-end
- ✅ Confirm your setup works before mainnet

This is a **valid Phase 0 validation** approach since you're testing the actual functionality, not relying on inactive testnet validators.

---

**Last Updated:** December 19, 2025  
**Status:** Ready to test with your own validator

