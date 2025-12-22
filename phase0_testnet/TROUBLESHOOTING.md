# Troubleshooting: No Validator Queries After 30+ Minutes

## Issue Identified ✅

**Root Cause:** Miner is NOT registered with NATIX application server!

There are **TWO separate registrations** needed:
1. ✅ **Bittensor Registration** - DONE (UID 88 on testnet)
2. ❌ **NATIX Application Server Registration** - MISSING!

Without NATIX registration, validators won't know to query your miner.

## Solution: Register with NATIX Application Server

### Step 1: Register with NATIX

Run this command from the `streetvision-subnet` directory:

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
./register.sh 88 testnet_wallet miner_hotkey miner natix-network-org/roadwork
```

**What this does:**
- Uses your UID: 88
- Wallet: testnet_wallet
- Hotkey: miner_hotkey
- Type: miner
- Model: natix-network-org/roadwork (official NATIX model)

### Step 2: Verify Registration

After running the script, it should:
1. Generate a timestamp
2. Sign it with your Bittensor hotkey
3. Send registration to NATIX server
4. Show confirmation

### Step 3: Keep Miner Running

The miner should continue running. After NATIX registration, validators should start querying within 5-30 minutes.

## What to Expect After Registration

```
✅ Registration succeeded for UID 88.
✅ Miner should start receiving queries soon.
```

Then in miner logs, you should see:
```
INFO | Received image challenge!
INFO | PREDICTION = 0.623
INFO | LABEL (testnet only) = 1
```

## Additional Checks

If still no queries after registration:

1. **Check Registration Status:**
   ```bash
   curl -s https://hydra.natix.network/participants/registration-status/88 | jq
   ```

2. **Verify Miner is Still Running:**
   ```bash
   ps aux | grep miner
   ```

3. **Check Network Connectivity:**
   - Ensure port 8091 is accessible
   - Check firewall settings

4. **Check Testnet Activity:**
   ```bash
   poetry run btcli subnet show --netuid 323 --network test
   ```
   Look for validators with emissions > 0 (active validators)

## Documentation Reference

From NATIX docs (Mining.md):
> "Once registered on-chain, you must also register on the **Natix application server**. make sure you've registered, and received your `uid` on Bittensor"

This is the missing step! 🔑

