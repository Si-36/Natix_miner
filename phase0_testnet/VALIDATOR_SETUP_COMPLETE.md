# ✅ Validator Setup Complete!

**Date:** December 19, 2025  
**Status:** Validator ready to test miner

---

## 🎉 What's Been Set Up

### 1. Validator Hotkey Created
- **Hotkey name:** `validator_hotkey`
- **SS58 address:** `5He5Ri1b2HTgBtDTX9YeA3BgcL3AurnmdaYoN7bpXamK1F6U`
- **Mnemonic:** `decline extend sign useless silly work problem size pitch cradle slush program`
- **⚠️ SAVE THIS MNEMONIC SECURELY!**

### 2. Validator Registered on Bittensor Testnet
- **Network:** test (subnet 323)
- **UID:** 89
- **Balance:** 9.9994 τ (testnet TAO)

### 3. Validator Registered with NATIX Dev Server
- **Registration status:** "pending" (enqueued for processing)
- **Role:** validator
- **UID:** 89
- **Endpoint:** `https://hydra.dev.natix.network`

### 4. Startup Script Created
- **File:** `start_validator_testnet.sh`
- **Simplified for testing** (no WandB/HuggingFace requirements)
- **WandB disabled** with `--wandb.off` flag

---

## 🚀 How to Start Testing

### Terminal 1: Start/Check Miner

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Check if miner is running
ps aux | grep "neurons/miner.py" | grep -v grep

# If not running, start it:
./start_miner.sh
```

**Miner details:**
- UID: 88
- Hotkey: `5DMBvP1JFVzpihTPUt22G97U3dGYw2kfRmiTMRLxuhYv6QBk`
- Port: 8091

### Terminal 2: Start Validator

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Start validator
./start_validator_testnet.sh
```

**Validator details:**
- UID: 89
- Hotkey: `5He5Ri1b2HTgBtDTX9YeA3BgcL3AurnmdaYoN7bpXamK1F6U`
- Port: 8092
- Proxy port: 10913

---

## 🔍 What to Expect

### Validator Behavior:
1. **Connects to testnet** (subnet 323)
2. **Loads metagraph** (sees all miners including UID 88)
3. **Sends queries every ~30 seconds** to random miners
4. **Your miner (UID 88) should receive queries**

### Miner Behavior:
1. **Receives image challenges** from validator
2. **Processes images** using ViT model on GPU
3. **Returns predictions** (0.0-1.0 range)
4. **Logs:** "Received image challenge!" and "PREDICTION = 0.xxxx"

### Expected Logs:

**Validator logs:**
```
INFO | Sending image challenge to X miners
INFO | Miner UIDs to provide with challenge: [88, ...]
INFO | Predictions of challenge: [0.554807, ...]
```

**Miner logs:**
```
INFO | Received image challenge!
INFO | PREDICTION = 0.554807
INFO | LABEL (testnet only) = 1
```

---

## 📊 Monitoring

### Check Both Are Connected:

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet

# Check metagraph
poetry run python -c "
import bittensor as bt
metagraph = bt.metagraph(netuid=323, network='test')
print(f'Miner UID 88 serving: {metagraph.axons[88].is_serving}')
print(f'Validator UID 89 serving: {metagraph.axons[89].is_serving}')
print(f'Total neurons: {metagraph.n}')
"
```

### Check If Queries Are Being Sent:

Watch the logs in both terminals. You should see:
- **Validator:** Sending queries every 30-60 seconds
- **Miner:** Receiving and responding to queries

---

## 🎯 Success Criteria

✅ **Validator starts without errors**  
✅ **Miner receives queries** ("Received image challenge!")  
✅ **Miner returns predictions** ("PREDICTION = 0.xxxx")  
✅ **Validator logs show responses** from UID 88  

If all of these happen, your setup is **fully validated** and ready for mainnet!

---

## ⚠️ Troubleshooting

### Validator Doesn't Query Miner:

1. **Check both are in metagraph:**
   ```bash
   poetry run btcli subnet show --netuid 323 --network test | grep -E "(88|89)"
   ```

2. **Check axons are serving:**
   - Miner logs should show: "Axon created" and serving on port 8091
   - Validator logs should show: "Axon created" and serving on port 8092

3. **Check validator can see miner:**
   - Validator should log available UIDs when querying
   - UID 88 should be in the list

### Miner Not Receiving Queries:

1. **Restart miner:**
   ```bash
   pkill -f "neurons/miner.py"
   ./start_miner.sh
   ```

2. **Check GPU is working:**
   ```bash
   poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Check miner registration:**
   ```bash
   poetry run btcli wallet overview --wallet.name testnet_wallet --subtensor.network test
   ```

---

## 📝 Summary of All Components

### Miner (UID 88)
- ✅ Registered on Bittensor testnet
- ✅ Registered with NATIX dev server (pending)
- ✅ Model tested locally (works!)
- ✅ GPU configured and working
- 🔄 Ready to receive queries from validator

### Validator (UID 89)
- ✅ Hotkey created
- ✅ Registered on Bittensor testnet
- ✅ Registered with NATIX dev server (pending)
- ✅ Startup script configured
- 🔄 Ready to query miners

### Next Action
1. **Start miner** (Terminal 1)
2. **Start validator** (Terminal 2)
3. **Watch logs** for queries and responses
4. **Verify end-to-end** functionality

---

**Setup completed:** December 19, 2025  
**Status:** ✅ Ready to test!  
**Time to start:** Now!

