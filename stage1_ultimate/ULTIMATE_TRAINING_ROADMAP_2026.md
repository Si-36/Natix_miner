# 🚀 ULTIMATE TRAINING ROADMAP 2026
**Complete Path to 99%+ MCC on NATIX Roadwork Detection**

**Version**: 1.0  
**Last Updated**: January 8, 2026  
**Status**: Production Ready

---

## 📋 EXECUTIVE SUMMARY

This roadmap provides a **complete, step-by-step plan** to upgrade `stage1_ultimate/` with the latest 2025/2026 training techniques to achieve **99%+ MCC** on NATIX Subnet 72 roadwork detection.

### 🎯 Goals
- **99%+ MCC accuracy** (Matthews Correlation Coefficient)
- **30× faster VLM training** (UnSloth + FlashAttention-3)
- **8 new models** (YOLO-Master, Qwen3-VL, ADFNet, etc.)
- **Active learning pipeline** (continuous improvement)
- **Production-ready deployment**

### 📊 Current State
- ✅ **Tier 0 Infrastructure**: DAG pipeline, artifact registry, split contracts
- ✅ **Sophia-H optimizer**: 2× faster than AdamW
- ✅ **GPS sampler**: EXISTS but not wired
- ✅ **Heavy augmentation**: EXISTS (Kornia)
- ❌ **Callbacks folder**: EMPTY
- ❌ **PEFT configs**: MISSING
- ❌ **Latest 2026 techniques**: NOT IMPLEMENTED

---

## 🗺️ IMPLEMENTATION TIMELINE

### **Week 0: Critical Gap Closure (18.5 hours)** 🔴 BLOCKING
**Status**: 🔴 MUST DO FIRST
**Duration**: 18.5 hours (was 16h - added 2.5h for critical verification)
**Document**: [`WEEK_0_EXECUTION_PLAN.md`](./WEEK_0_EXECUTION_PLAN.md)

**⚠️ UPDATED**: Added 5 critical verification steps to prevent silent failures!

#### Tasks:
1. **Day 0.1**: Wire GPS-aware sampling (4.5h)
   - **NEW**: Verify GPS schema (latitude/longitude vs string)
   - GPS sampler EXISTS but not wired
   - Update `configs/data/natix.yaml`
   - Wire into `src/data/datamodule.py`
   - **NEW**: Verify split ratios match spec
   - **Impact**: +7-10% MCC (BIGGEST WIN!)

2. **Day 0.2**: Create latest augmentations (5h)
   - Create `src/data/augmentation/latest_aug_2025.py`
   - TrivialAugment + CutMix + MixUp
   - **NEW**: Wire into datamodule (integration!)
   - **Impact**: +3-5% detection accuracy

3. **Day 0.3**: Implement callbacks (5h)
   - Create `src/training/callbacks/mcc_callback.py`
   - Create `src/training/callbacks/ema_callback.py`
   - **NEW**: Register callbacks in trainer (integration!)
   - **Impact**: Track MCC, model stability

4. **Day 0.4**: Create PEFT config stubs (4h)
   - Create `src/training/lora/adalora_config.py`
   - Create `src/training/lora/vera_config.py`
   - Create `src/training/lora/ia3_config.py`
   - **Impact**: Unblocks Week 1.5

**✅ Success Criteria** (UPDATED):
- GPS schema verified (latitude/longitude floats OR normalized)
- Dataset mode chosen (Mode A or Mode B)
- Split ratios verified (match TRAINING_PLAN spec)
- GPS sampler wired + tested
- Latest augmentation created + **integrated** + tested
- MCC + EMA callbacks created + **registered** + tested
- PEFT configs created + tested
- All imports work
- No errors in dry-run training

**📚 See Also**: [`WEEK_0_CRITICAL_FIXES.md`](./WEEK_0_CRITICAL_FIXES.md) for details on what was fixed

---

### **Week 1: Core Training Infrastructure (40 hours)**
**Status**: ⏳ AFTER WEEK 0  
**Duration**: 10 days × 4 hours  
**Reference**: `TRAINING_PLAN_2026_CLEAN.md` lines 335-349

#### Components:
- **UnSloth Trainer**: 30× faster VLM training
- **LoRA/QLoRA Trainer**: Fine-tune 70B+ models
- **DPO Trainer**: Alignment training
- **Active Learning**: Sample hard examples
- **VL2Lite Distillation**: +7% accuracy
- **MCC/EMA Callbacks**: Already done in Week 0!

**Expected Impact**: 30× faster training, 70B+ model support

---

### **Week 1.5: Latest 2025/2026 Techniques (40 hours)** 🔥 CRITICAL
**Status**: ⏳ AFTER WEEK 1  
**Duration**: 10 days × 4 hours  
**Reference**: `TRAINING_PLAN_2026_CLEAN.md` lines 352-411

#### Breakthrough Techniques:
1. **DAPO (GRPO++)**: +67% AIME improvement (30% → 50%)
   - 4 critical fixes to vanilla GRPO
   - Stable RL training
   - **Impact**: AIME 30% → 50%

2. **Advanced PEFT**: AdaLoRA, VeRA, IA³
   - AdaLoRA: +2-3% accuracy over LoRA
   - VeRA: 99% fewer parameters
   - IA³: 0.01% trainable params
   - **Impact**: +2-3% accuracy, 99% fewer params

3. **FlashAttention-3**: 1.5-2× faster than FA2
   - FP8 support for H100
   - **Impact**: 1.5-2× faster

4. **Modern Optimizers**:
   - AdEMAMix (VLM fine-tuning)
   - Muon+AdamW hybrid (HF-style loops)
   - Ultralytics `optimizer="auto"` (detectors)
   - Schedule-Free AdamW (no LR schedule)
   - **Impact**: Stable, fast convergence

5. **TrivialAugment**: Zero hyperparameters
   - CutMix: +3.5% object detection
   - MixUp: +2.3% classification
   - **Impact**: +3-5% detection

**Expected Impact**: +67% AIME, +2-3% accuracy, 1.5-2× faster

---

### **Week 2: New Model Implementations (40 hours)**
**Status**: ⏳ AFTER WEEK 1.5  
**Duration**: 10 days × 4 hours  
**Reference**: `TRAINING_PLAN_2026_CLEAN.md` lines 1438-1600

#### Models to Implement:
1. **YOLO-Master**: 62.5% mAP (SOTA detection)
2. **Qwen3-VL**: 4B/8B/32B/72B (VLM reasoning)
3. **ADFNet**: Night specialist
4. **Depth Anything 3**: Depth estimation
5. **InternVL3**: 78B VLM
6. **Llama 4 Vision**: 90B VLM
7. **Molmo-2**: 72B VLM
8. **GLM-4.6V**: 9B VLM

**Expected Impact**: 8 new models, 99%+ MCC

---

### **Week 3: Advanced Training Techniques (40 hours)**
**Status**: ⏳ AFTER WEEK 2  
**Duration**: 10 days × 4 hours  
**Reference**: `TRAINING_PLAN_2026_CLEAN.md` lines 1602-1800

#### Techniques:
- Knowledge distillation (VL2Lite)
- Multi-task learning
- Self-training
- Pseudo-labeling (SAM 3)
- Test-time augmentation

**Expected Impact**: +5-7% accuracy

---

## 📊 EXPECTED RESULTS

### Accuracy Targets
| Metric | Baseline | After Week 0 | After Week 1.5 | After Week 3 |
|--------|----------|--------------|----------------|--------------|
| **MCC** | 85% | 92-95% | 96-98% | **99%+** |
| **Accuracy** | 88% | 93-95% | 96-98% | **99%+** |
| **FNR** | 15% | 8-10% | 5-7% | **<5%** |

### Training Speed
| Component | Baseline | After Week 1.5 |
|-----------|----------|----------------|
| **VLM Training** | 1× | **30×** (UnSloth + FA3) |
| **Convergence** | 1× | **2×** (Sophia-H + AdEMAMix) |

---

## 🔑 KEY DECISIONS

### Optimizer Decision Tree
| What you're training | Use | Why |
|---------------------|-----|-----|
| **VLM fine-tune** | AdEMAMix | Stable VLM fine-tuning |
| **Ultralytics detectors** | `optimizer="auto"` | Optimized defaults |
| **Stage1 backbone** | Keep Sophia-H | Avoid churn |
| **HF-style loops** | Muon+AdamW hybrid | Stability + speed |

### PEFT Decision Tree
| Model Size | Use | Why |
|-----------|-----|-----|
| **<4B** | Standard LoRA | Simple, effective |
| **4-32B** | AdaLoRA | +2-3% accuracy |
| **32-70B** | VeRA | 99% fewer params |
| **70B+** | IA³ | 0.01% trainable params |

---

## 📚 REFERENCE DOCUMENTS

### Core Documents
1. **[TRAINING_PLAN_2026_CLEAN.md](../TRAINING_PLAN_2026_CLEAN.md)**: Complete training plan
2. **[INFERENCE_ARCHITECTURE_2026.md](../INFERENCE_ARCHITECTURE_2026.md)**: Inference deployment
3. **[ULTIMATE_PLAN_2026_LOCAL_FIRST.md](../ULTIMATE_PLAN_2026_LOCAL_FIRST.md)**: Overall architecture

### Week-Specific Documents
1. **[WEEK_0_EXECUTION_PLAN.md](./WEEK_0_EXECUTION_PLAN.md)**: Week 0 detailed plan

---

## 🚀 GETTING STARTED

### Step 1: Read Week 0 Plan
```bash
cat stage1_ultimate/WEEK_0_EXECUTION_PLAN.md
```

### Step 2: Run Pre-Flight Checks
```bash
cd /home/sina/projects/miner_b/stage1_ultimate
python3 --version  # Should be 3.11+
ls -la src/data/samplers/gps_weighted_sampler.py  # Should exist
ls -la src/training/callbacks/  # Should be empty
```

### Step 3: Start Week 0
Follow `WEEK_0_EXECUTION_PLAN.md` step-by-step.

---

**🎯 GOAL**: 99%+ MCC on NATIX Roadwork Detection  
**⏱️ TIMELINE**: 178.5 hours (4.5 weeks) - Updated from 176h
**🔥 PRIORITY**: Week 0 is BLOCKING - start there!

**⚠️ UPDATED**: Week 0 now 18.5h (was 16h) - added critical verification steps!


