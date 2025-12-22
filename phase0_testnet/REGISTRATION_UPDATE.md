# ✅ NATIX Registration Updated to Dev Endpoint

**Date:** December 19, 2025  
**Status:** Registration successful with dev endpoint

---

## 🔄 Changes Made

### 1. Updated Registration Endpoint

**Changed from:** `https://hydra.natix.network`  
**Changed to:** `https://hydra.dev.natix.network` (dev/testnet endpoint)

### 2. Updated Files

**`register_natix.py`:**
- ✅ Changed `BASE_URL` to `https://hydra.dev.natix.network`
- ✅ Using endpoint: `/participants/register` (plural)
- ✅ Status check: `/participants/registration-status/{UID}`

**`miner.env`:**
- ✅ Updated `PROXY_CLIENT_URL=https://hydra.dev.natix.network`

**`validator.env`:**
- ✅ Created new file with testnet configuration
- ✅ Set `PROXY_CLIENT_URL=https://hydra.dev.natix.network`

---

## ✅ Registration Result

**Miner Registration (UID 88):**
```
✅ Registration successful!
{
  "detail": "Registration enqueued and will be processed shortly."
}

Status: "pending"
Hotkey: 5DMBvP1JFVzpihTPUt22G97U3dGYw2kfRmiTMRLxuhYv6QBk
Role: "miner"
UID: 88
```

**Note:** Status shows "pending" which is normal. Registration is enqueued and will be processed.

---

## 🎯 Next Steps

### Option 1: Run Your Own Validator (Recommended)

Since NATIX focuses on mainnet, run your own validator to test your miner:

1. **Create validator hotkey:**
   ```bash
   poetry run btcli wallet new_hotkey \
     --wallet.name testnet_wallet \
     --wallet.hotkey validator_hotkey \
     --subtensor.network test
   ```

2. **Register validator on testnet:**
   ```bash
   poetry run btcli subnet register \
     --netuid 323 \
     --wallet.name testnet_wallet \
     --wallet.hotkey validator_hotkey \
     --subtensor.network test
   ```

3. **Register validator with NATIX:**
   - Update `register_natix.py` with validator UID
   - Change `type` to `"validator"` in the payload
   - Run: `poetry run python register_natix.py`

4. **Start validator:**
   ```bash
   ./start_validator.sh
   ```

**Full guide:** See `SETUP_VALIDATOR_FOR_TESTING.md`

### Option 2: Wait for Approval

The registration is pending. It may be approved automatically or require manual review. Check status:

```bash
cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
curl -s "https://hydra.dev.natix.network/participants/registration-status/88" | jq
```

---

## 📝 Summary

✅ **Miner registered with dev endpoint**  
✅ **PROXY_CLIENT_URL updated in miner.env**  
✅ **Validator.env created for testnet**  
✅ **Registration shows "pending" status (normal)**

**Recommended action:** Set up your own validator to test your miner end-to-end.

---

**Last Updated:** December 19, 2025

