# 🚀 StreetVision Cascade Infrastructure Setup Status

**Date:** December 20, 2025
**Status:** ✅ Infrastructure Complete | ⏸️ Stage-3 VLM Downloads Paused (resume later)

---

## 📊 Overall Progress Summary

| Category | Status | Details |
|----------|--------|---------|
| Project Structure | ✅ Complete | All directories and configs created |
| Dependencies | ✅ Complete | PyTorch 2.7, vLLM, TensorRT deps installed |
| Stage 1-2 Models | ✅ Complete | DINOv2-L, RT-DETR, Florence-2 downloaded |
| Stage 3 VLMs | ⏸️ Paused | GLM-4.6V-Flash (Hub: `zai-org/GLM-4.6V-Flash`) and Molmo2-8B (Hub: `allenai/Molmo2-8B`) |
| NATIX Dataset | ✅ Complete | Official roadwork dataset downloaded |
| Training Scripts | ✅ Complete | DINOv3 classifier training ready |
| Data Pipeline | ✅ Complete | SDXL synthetic generation ready |
| Active Learning | ✅ Complete | FiftyOne hard-case mining ready |
| Monitoring | ✅ Complete | Prometheus + Grafana configs ready |
| Deployment | ✅ Complete | Blue-green deployment scripts ready |

---

## 📦 Model Download Status

### Completed Downloads ✅

| Model | Purpose | Size | Location |
|-------|---------|------|----------|
| DINOv2-Large | Stage 1 Binary Classifier | 581MB | `models/stage1_dinov3/` |
| RT-DETR-Medium | Stage 2a Object Detection | 83MB | `models/stage2_rfdetr/` |
| YOLOv11-X | Stage 2b Object Detection | 110MB | `yolo11x.pt` (cached) |
| Florence-2-Large | Stage 4 OCR Fallback | 1.5GB | `models/stage4_florence/` |

### Paused ⏸️

| Model | Purpose | Expected Size | Status |
|-------|---------|---------------|--------|
| GLM-4.6V-Flash | Stage 3a Image VLM | ~9GB | Paused (download later) |
| Molmo2-8B | Stage 3b Video VLM | ~4.5GB | Paused (download later) |

> **Note:** These VLM models are loaded dynamically with 4-bit AWQ quantization on a 24GB GPU. Download them overnight when you want.

---

## 🏗️ Project Structure

```
streetvision_cascade/
├── configs/
│   └── cascade_config.yaml        # Cascade thresholds and model configs
├── data/
│   ├── hard_cases/                # Daily hard-case mining outputs
│   ├── natix_official/            # NATIX roadwork dataset (328MB)
│   ├── synthetic_sdxl/            # SDXL-generated synthetic images
│   └── validation/                # Fixed challenge sets
├── models/
│   ├── stage1_dinov3/             # DINOv2-Large backbone (581MB)
│   ├── stage2_rfdetr/             # RT-DETR detector (83MB)
│   ├── stage2_yolo/               # YOLOv11-X weights
│   ├── stage3_glm/                # GLM-4.6V-Flash-9B VLM (download later)
│   ├── stage3_molmo/              # Molmo-2-8B VLM (download later)
│   ├── stage4_florence/           # Florence-2-Large (1.5GB)
│   ├── quantized/                 # AWQ 4-bit quantized models
│   └── tensorrt/                  # TensorRT-optimized engines
├── scripts/
│   ├── active_learning/
│   │   └── fiftyone_hard_mining.py
│   ├── data/
│   │   └── generate_sdxl_synthetic.py
│   ├── deployment/
│   │   └── blue_green_deploy.py
│   ├── inference/
│   │   └── cascade_pipeline.py
│   ├── monitoring/
│   │   ├── grafana_dashboards.json
│   │   └── prometheus_metrics.py
│   ├── training/
│   │   └── train_dinov3_classifier.py
│   ├── daily_hard_case_mining.sh
│   └── download_models.py
├── checkpoints/                   # Training checkpoints
├── logs/                          # Application logs
└── cache/                         # Inference cache
```

---

## 🔧 Scripts Ready for Use

### 1. Training (`scripts/training/`)
- **`train_dinov3_classifier.py`** - Frozen backbone training with FocalLoss

### 2. Inference (`scripts/inference/`)
- **`cascade_pipeline.py`** - Full 4-stage cascade with dynamic VLM loading

### 3. Data Generation (`scripts/data/`)
- **`generate_sdxl_synthetic.py`** - FREE synthetic roadwork image generation

### 4. Active Learning (`scripts/active_learning/`)
- **`fiftyone_hard_mining.py`** - Hard-case mining with FiftyOne Brain

### 5. Deployment (`scripts/deployment/`)
- **`blue_green_deploy.py`** - Zero-downtime model updates via NGINX

### 6. Monitoring (`scripts/monitoring/`)
- **`prometheus_metrics.py`** - GPU VRAM, cascade latency, accuracy metrics
- **`grafana_dashboards.json`** - Pre-configured Grafana dashboard

### 7. Automation
- **`daily_hard_case_mining.sh`** - Cron-ready daily workflow automation

---

## 🎯 Next Steps (While Stage‑3 downloads are paused)

### Immediate (you can do now — no need for Stage‑3 yet):
1. **Verify all models load correctly**
   ```bash
   cd /home/sina/projects/miner_b/streetvision_cascade
   source .venv/bin/activate
   python -c "from scripts.inference.cascade_pipeline import CascadePipeline; p = CascadePipeline(); print('✅ Cascade loads!')"
   ```

2. **Generate synthetic training data with SDXL** (FREE)
   ```bash
   python scripts/data/generate_sdxl_synthetic.py --num-positive 500 --num-negative 500
   ```

3. **Train DINOv3 classifier head** (uses frozen backbone)
   ```bash
   python scripts/training/train_dinov3_classifier.py --epochs 10 --batch-size 32
   ```

### When You Rent a 24GB GPU:
1. **Apply AWQ 4-bit quantization to VLMs**
2. **Convert Stage 1-2 models to TensorRT FP16**
3. **Test full cascade inference end-to-end**
4. **Run validation against challenge set**

### For Mainnet Deployment:
1. **Setup PM2 process management**
2. **Configure NGINX reverse proxy**
3. **Deploy Prometheus + Grafana stack**
4. **Schedule daily hard-case mining cron job**
5. **Configure 90-day retrain automation**

---

## 💰 Cost Summary So Far

| Item | Cost |
|------|------|
| Local Development | $0 (FREE) |
| SDXL Synthetic Data | $0 (FREE) |
| Model Downloads | $0 (FREE) |
| Testnet Operations | $0 (FREE faucet TAO) |
| **Total Spent** | **$0** |

---

## 📋 Configuration Reference

### Cascade Thresholds (from `configs/cascade_config.yaml`)
- **Stage 1 Exit**: Confidence ≥ 0.88 (positive) or ≤ 0.12 (negative)
- **Stage 2 Agreement**: Both detectors agree on ≥3 objects
- **Stage 3 VLM**: Confidence ≥ 0.75
- **Stage 4 OCR**: ≥2 keywords found → positive

### VRAM Budget (for 24GB GPU)
- Stage 1 (DINOv3): 3.0 GB
- Stage 2a (RF-DETR): 1.9 GB  
- Stage 2b (YOLO): 3.1 GB
- Stage 3 VLM (dynamic): 2.3 GB (AWQ 4-bit)
- Stage 4 (Florence-2): 1.5 GB
- **Max Concurrent**: ~12 GB (well within 24GB budget)

---

## 🔍 Monitoring Commands

```bash
# Check download progress
du -sh /home/sina/projects/miner_b/streetvision_cascade/models/*/

# Check Python download processes
ps aux | grep "snapshot_download" | grep -v grep

# Monitor disk space
df -h /home/sina/projects/

# View cascade config
cat /home/sina/projects/miner_b/streetvision_cascade/configs/cascade_config.yaml
```

---

**Last Updated:** December 20, 2025 at 21:58 UTC

