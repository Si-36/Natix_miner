# 💬 DISCORD TESTNET TAO REQUEST

## 📋 Instructions

1. **Join Bittensor Discord:** https://discord.gg/bittensor
2. **Navigate to:** #testnet-faucet channel
3. **Copy and paste the message below:**

---

## 📝 Request Message (Copy This)

```
Request testnet TAO for miner testing on netuid 323 (NATIX StreetVision subnet).

Wallet Address: 5H8deNTX8atqyMvxufb24CoGLY7nCYBC16x5hFdJzLPQPAP2

Purpose: Phase 0 validation testing for roadwork detection miner before mainnet deployment. Will be testing model performance, latency optimization, and subnet mechanics.

Thank you!
```

---

## ⏱️ Expected Timeline

- **Request submitted:** (Fill in after posting)
- **Typical wait time:** 24-48 hours
- **Check balance:** `poetry run btcli wallet balance --wallet.name testnet_wallet --network test`

---

## ✅ After Receiving TAO

Once you have testnet TAO (usually 100+ test TAO):

1. **Verify balance:**
   ```bash
   poetry run btcli wallet balance --wallet.name testnet_wallet --network test
   ```

2. **Check subnet info:**
   ```bash
   poetry run btcli subnet show --netuid 323 --network test
   ```

3. **Register on testnet:**
   ```bash
   poetry run btcli subnet register \
     --netuid 323 \
     --wallet.name testnet_wallet \
     --wallet.hotkey miner_hotkey \
     --network test
   ```

---

## 📊 Your Wallet Info

- **Wallet Name:** testnet_wallet
- **Hotkey Name:** miner_hotkey
- **Wallet Address:** 5H8deNTX8atqyMvxufb24CoGLY7nCYBC16x5hFdJzLPQPAP2
- **Network:** test (testnet)
- **Target Subnet:** netuid 323 (NATIX StreetVision)
- **Current Balance:** 0.0 τ (awaiting Discord request)

---

## 🔗 Useful Links

- **Bittensor Discord:** https://discord.gg/bittensor
- **Testnet Explorer:** https://taostats.io/testnet/
- **NATIX Subnet GitHub:** https://github.com/natixnetwork/streetvision-subnet
- **Bittensor Docs:** https://docs.learnbittensor.org/

---

**Status:** ⏳ Awaiting Discord TAO request
