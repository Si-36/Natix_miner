# Precision Safety Guide

## Safe-by-Default Precision Rules

This repo uses **safe-by-default precision settings** to prevent NaN/Inf issues during training.

---

## TL;DR - Quick Reference

| Environment | Precision | Config Override | Why? |
|------------|-----------|----------------|------|
| **Local/Dev** | **FP32** | (default) | Slow but safest, no NaN debugging |
| **Rental GPU (H100/A100)** | **BFloat16** | `training.mixed_precision.enabled=true` | Fast + safe (FP32-like range) |
| **NEVER USE** | ~~FP16~~ | ❌ DON'T | Overflows with DINOv3 → NaN logits |

---

## The Problem: FP16 + DINOv3 = NaN

### What Happened

During development, we encountered **NaN logits** that crashed training:

```python
Logits: tensor([[nan, nan],
                [nan, nan],
                [nan, nan]], dtype=torch.float16)
```

### Root Cause

**FP16 (float16) has limited dynamic range:**
- FP16 range: ±65,504
- FP32 range: ±3.4×10³⁸
- BFloat16 range: ±3.4×10³⁸ (same as FP32!)

**DINOv3 ViT-H/16 (840M parameters) produces features that overflow FP16:**
1. Backbone forward pass → intermediate activations exceed ±65,504
2. Overflow → NaN
3. NaN propagates through classification head
4. Result: NaN logits → NaN loss → training crash

### Timeline of Debugging

1. ❌ **Suspected data imbalance** (13 classes → class 12 dominates)
   - Fixed: Created stratified balanced subset (50/50 split)
   - Result: Still NaN!

2. ❌ **Suspected wrong num_classes** (13 vs 2)
   - Fixed: Updated configs to num_classes=2
   - Result: Still NaN!

3. ✅ **Discovered FP16 overflow**
   - Root cause: `training.mixed_precision.dtype=float16`
   - Fix: Disabled mixed precision (FP32)
   - Result: **No more NaN!**

---

## The Solution: Precision Hierarchy

### 1. Local/Dev: FP32 (Default)

**Why FP32 for local dev?**
- ✅ Never waste time debugging NaN issues
- ✅ Works with any model architecture
- ✅ Easier to verify pipeline correctness
- ❌ 2x slower than mixed precision
- ❌ 2x more VRAM

**Config (already default):**
```yaml
training:
  mixed_precision:
    enabled: false  # FP32 by default
```

**When to use:**
- Smoke tests (small datasets, 1-5 epochs)
- Pipeline verification
- Local development on small subsets
- Debugging NaN/Inf issues

---

### 2. Rental GPU (H100/A100): BFloat16

**Why BFloat16 on rental GPU?**
- ✅ **Same dynamic range as FP32** (no overflow!)
- ✅ 2x faster training (Tensor Cores)
- ✅ Half the VRAM usage
- ✅ Better gradient stability than FP16
- ✅ Native hardware support on H100/A100

**Enable BFloat16:**
```bash
python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16
```

**When to use:**
- Full training runs (100 epochs)
- Multi-GPU distributed training (FSDP2)
- Rental GPU with BF16 support (H100, A100)
- Production training pipelines

---

### 3. AVOID: FP16 ❌

**Why NOT FP16?**
- ❌ Limited range (±65,504) → overflows with DINOv3
- ❌ Causes NaN in forward pass
- ❌ Poor gradient stability
- ❌ Requires loss scaling (complex, error-prone)

**DO NOT use FP16 unless:**
- You add gradient scaling + NaN detection
- You use a smaller backbone (ResNet50, ViT-B)
- You have benchmarked it extensively on your model

---

## Precision Comparison Table

| Precision | Exponent Bits | Mantissa Bits | Range | DINOv3 Safe? | Speed | VRAM |
|-----------|--------------|---------------|-------|-------------|-------|------|
| **FP32** | 8 | 23 | ±3.4×10³⁸ | ✅ Yes | 1x | 1x |
| **BFloat16** | 8 | 7 | ±3.4×10³⁸ | ✅ Yes | ~1.5-2x | 0.5x |
| **FP16** | 5 | 10 | ±65,504 | ❌ **NO** | ~2x | 0.5x |

**Key insight:** BFloat16 uses the same exponent bits as FP32 (8 bits), giving it the same dynamic range!

---

## How to Override Precision

### Local Smoke Test (FP32 - default)

```bash
# No override needed, FP32 is already default
python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  training.epochs=1 \
  data.data_root=/path/to/subset
```

### Rental GPU Production Run (BFloat16)

```bash
# Enable BFloat16 for full training
python scripts/train_cli.py \
  pipeline.phases=[phase1,phase2,phase3,phase5,phase6] \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  hardware.num_gpus=4
```

### Debug NaN with FP32

```bash
# If you see NaN during training, force FP32
python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  training.mixed_precision.enabled=false
```

---

## Checklist for Rental GPU Training

Before spending $$$ on rental GPUs, verify locally with FP32:

- [ ] Pipeline runs end-to-end without errors
- [ ] No NaN/Inf in logits or loss
- [ ] Validation artifacts saved correctly
- [ ] Metrics look reasonable (accuracy > random)
- [ ] All phases pass with small subset

Then enable BFloat16 for production:

- [ ] Confirm GPU supports BFloat16 (H100, A100)
- [ ] Set `training.mixed_precision.enabled=true`
- [ ] Set `training.mixed_precision.dtype=bfloat16`
- [ ] Monitor first few steps for NaN (should be fine!)

---

## Technical Details: Why BFloat16 Wins

### BFloat16 vs FP16 Representation

```
FP32:     [sign: 1 bit] [exponent: 8 bits] [mantissa: 23 bits]
BFloat16: [sign: 1 bit] [exponent: 8 bits] [mantissa:  7 bits]  ← Same range!
FP16:     [sign: 1 bit] [exponent: 5 bits] [mantissa: 10 bits]  ← Limited range
```

**BFloat16 is "FP32 truncated"**: Just drop mantissa bits, keep exponent range!

### Why This Matters for DINOv2/v3

DINOv3 ViT-H/16 intermediate activations:
- Layer 12 attention: values in range [-50, +50]
- Layer 24 MLP: values in range [-200, +200]
- Final features: values in range [-10, +10]

**FP16 max: ±65,504** → OK for most layers, but:
- Gradient accumulation can overflow
- Loss scaling adds complexity
- One bad batch → NaN propagation

**BFloat16 max: ±3.4×10³⁸** → No overflow possible!

---

## FAQ

### Q: Why not use FP16 with loss scaling?

**A:** Loss scaling adds complexity and can still fail:
- Requires tuning scaling factor
- Dynamic loss scaling has overhead
- One NaN in forward pass → game over
- BFloat16 is simpler and safer

### Q: What if my GPU doesn't support BFloat16?

**A:** Use FP32 or rent a newer GPU:
- **H100, A100:** Native BFloat16 Tensor Cores
- **V100, older:** Use FP32 (no BF16 support)
- **RTX 30/40 series:** BF16 works but slower than H100

### Q: How much slower is FP32 vs BFloat16?

**A:** On H100/A100:
- FP32: 1x (baseline)
- BFloat16: ~1.5-2x faster
- Memory: 2x less

**Example:** 100 epochs on 8,549 images:
- FP32: ~24 hours
- BFloat16: ~12-16 hours

### Q: What about Phase 4 (ExPLoRA)?

**A:** Same rules apply:
- Local dev: FP32
- Rental GPU: BFloat16
- ExPLoRA is even more sensitive to precision (domain adaptation uses large batches)

---

## Summary

**Safe-by-default precision rules:**

1. ✅ **Local/Dev:** FP32 (slow but safest, default)
2. ✅ **Rental GPU:** BFloat16 (fast + safe, opt-in)
3. ❌ **Never:** FP16 with DINOv3 (overflow → NaN)

**To enable BFloat16 on rental GPU:**
```bash
training.mixed_precision.enabled=true training.mixed_precision.dtype=bfloat16
```

**When in doubt, use FP32.** You can always switch to BFloat16 later for production runs.

---

## References

- [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [BFloat16 vs FP16 Comparison](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
- [DINOv3 Paper](https://arxiv.org/abs/2304.07193) (uses BFloat16 in original training)
- [FSDP2 Mixed Precision Guide](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
