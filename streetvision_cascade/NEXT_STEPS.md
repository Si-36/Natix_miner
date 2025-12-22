# 📋 Next Steps Based on REALISTIC_DEPLOYMENT_PLAN.md

**Current Status:** ✅ All 6 models downloaded
**Next Phase:** Model Optimization & Training (Days 2-4)

---

## 🎯 What You Should Do Next (In Order)

### ✅ **PHASE 1: Model Download** - COMPLETE
- [x] DINOv3-vith16plus (3.36GB)
- [x] RF-DETR, YOLOv12
- [x] GLM-4.6V-Flash (20GB)
- [x] Molmo-2-8B (33GB)
- [x] Florence-2

---

### 🔄 **PHASE 2: Model Optimization (Day 2)** - **DO THIS NEXT**

#### 2.1 Quantize VLMs (4-bit AWQ)
**Goal:** Reduce VRAM usage for GLM-4.6V and Molmo-2-8B

```bash
# Quantize GLM-4.6V-Flash (9GB → ~2.3GB)
python scripts/quantize_glm.py \
  --model_path models/stage3_glm/GLM-4.6V-Flash \
  --output_path models/stage3_glm/GLM-4.6V-Flash-4bit \
  --quantize_method awq

# Quantize Molmo-2-8B (4.5GB → ~1.2GB)
python scripts/quantize_molmo.py \
  --model_path models/stage3_molmo/Molmo2-8B \
  --output_path models/stage3_molmo/Molmo2-8B-4bit \
  --quantize_method awq
```

**Why:** Fits all models in 24GB VRAM (essential for RTX 3090/4090)

#### 2.2 Convert to TensorRT FP16
**Goal:** 2-3× speedup for inference

```bash
# Convert DINOv3 backbone
python scripts/export_tensorrt_dinov3.py \
  --model_path models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m \
  --output_path models/stage1_dinov3/dinov3-vith16plus-trt \
  --precision fp16

# Convert RF-DETR and YOLOv12
python scripts/export_tensorrt_detectors.py \
  --models_dir models/stage2_rfdetr \
  --output_dir models/stage2_rfdetr-trt \
  --precision fp16
```

**Why:** Faster inference = more queries handled = higher rewards

---

### 🎓 **PHASE 3: Training (Days 3-4)** - **DO AFTER OPTIMIZATION**

#### 3.1 Download NATIX Official Dataset (FREE)
```bash
# Download 8,000 roadwork images from HuggingFace
python scripts/download_natix_dataset.py \
  --output_dir data/natix_official \
  --split train  # 6,000 images
  --split val    # 2,000 images
```

#### 3.2 Generate SDXL Synthetic Data (FREE)
**Goal:** 1,000 synthetic images to boost training data

```bash
# Generate 500 positive (roadwork) + 500 negative (no roadwork) images
python scripts/generate_sdxl_synthetics.py \
  --output_dir data/sdxl_synthetic \
  --count 1000 \
  --prompts_file configs/sdxl_prompts.yaml
```

**Why:** FREE synthetic data (saves $40 vs Cosmos). Quality is good enough for initial training.

#### 3.3 Train DINOv3 Classifier Head
**Goal:** Frozen backbone training (only 300K params, 20× faster)

```bash
# Train classifier head (6-8 hours on RTX 3090)
python scripts/training/train_dinov3_classifier.py \
  --config configs/cascade_config.yaml \
  --train_data data/natix_official/train data/sdxl_synthetic/positive data/sdxl_synthetic/negative \
  --val_data data/natix_official/val \
  --output_dir checkpoints/dinov3-classifier-v1 \
  --epochs 10 \
  --batch_size 32 \
  --freeze_backbone true  # CRITICAL: Freeze 1.3B params, only train 300K head
```

**Target:** 96%+ validation accuracy

#### 3.4 Calibrate Cascade Thresholds
**Goal:** Tune exit thresholds for 60% Stage-1 exit rate

```bash
# Calibrate thresholds on validation set
python scripts/calibrate_cascade.py \
  --model_path checkpoints/dinov3-classifier-v1 \
  --val_data data/natix_official/val \
  --target_exit_rate 0.60 \
  --output_config configs/cascade_config_calibrated.yaml
```

---

### 🧪 **PHASE 4: Local Testing (Before Mainnet)** - **DO AFTER TRAINING**

#### 4.1 Test Cascade Pipeline Locally
```bash
# Test end-to-end inference
python scripts/inference/test_cascade.py \
  --config configs/cascade_config_calibrated.yaml \
  --test_images data/natix_official/val/images/ \
  --output_dir results/local_test \
  --benchmark true  # Measure latency, accuracy
```

**Check:**
- Latency: < 25ms for Stage 1, < 100ms end-to-end
- Accuracy: 96%+ on validation set
- Exit rate: ~60% exit at Stage 1

#### 4.2 Test with Sample Queries
```bash
# Simulate validator queries
python scripts/test_miner_response.py \
  --model_config configs/cascade_config_calibrated.yaml \
  --query_file test_queries.json \
  --output_file test_responses.json
```

---

### 🚀 **PHASE 5: Testnet Deployment (Optional Before Mainnet)**

**Note:** You mentioned testing on testnet first - this is smart!

```bash
# Deploy to testnet (if available)
# 1. Register on testnet
btcli subnet register --netuid 72 --wallet.name main_wallet --wallet.hotkey main_hotkey --subtensor.network test

# 2. Start testnet miner
python miner.py --netuid 72 --subtensor.network test --config configs/cascade_config_calibrated.yaml
```

**Test for:**
- Query handling
- Response latency
- Accuracy on real queries
- Stability (no crashes)

---

### 💰 **PHASE 6: Mainnet Deployment (Day 5 - AFTER TESTING)**

**Only after:**
- ✅ Local testing passed
- ✅ Testnet testing passed (if used)
- ✅ You have TAO (0.7 TAO for registration)
- ✅ You have GPU rented (RTX 3090/4090)
- ✅ NATIX registration approved

```bash
# Register on mainnet
btcli subnet register --netuid 72 --wallet.name main_wallet --wallet.hotkey main_hotkey

# Deploy 1 miner (conservative start)
python miner.py --netuid 72 --config configs/cascade_config_calibrated.yaml
```

---

## 📊 Current Status Summary

| Phase | Status | Next Action |
|-------|--------|-------------|
| Model Download | ✅ Complete | Move to Optimization |
| Model Optimization | ⏸️ Not Started | **DO THIS NOW** |
| Training | ⏸️ Not Started | After optimization |
| Local Testing | ⏸️ Not Started | After training |
| Testnet | ⏸️ Not Started | After local testing |
| Mainnet | ⏸️ Not Started | After testnet (or skip if confident) |

---

## 🎯 **IMMEDIATE NEXT STEP**

**Start with Model Optimization (Phase 2):**

1. **Quantize VLMs** (GLM & Molmo) to 4-bit AWQ
2. **Convert to TensorRT** (DINOv3, RF-DETR, YOLOv12)
3. **Verify VRAM budget** (should be ~21GB / 24GB total)

**Time:** 4-6 hours  
**Cost:** GPU rental time (~$7-10)  
**Result:** Optimized models ready for training

---

## 💡 **Key Decisions from Your Plan**

- ✅ **Start with 1 miner** (not 3) - conservative approach
- ✅ **Use SDXL for synthetic data** (FREE, not $40 Cosmos)
- ✅ **Frozen backbone training** (fast, cheap)
- ✅ **Test locally first** (smart!)
- ✅ **Testnet before mainnet** (lower risk)

---

**Ready to start Phase 2 (Optimization)?** This is the next logical step after model downloads are complete.

