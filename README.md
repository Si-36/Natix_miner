# Bittensor Subnet 72 (NATIX StreetVision) Miner

Professional mining operation for roadwork detection on dashcam imagery using a 6-stage cascade architecture.

## Project Overview

**Subnet:** Bittensor Subnet 72 - NATIX StreetVision
**Task:** Roadwork detection (binary classification from dashcam images/video)
**Architecture:** 6-model cascade (DINOv3 → RF-DETR + YOLO → GLM + Molmo → Florence)
**Target Accuracy:** 98%+ (Top 5 miner ranking)

## Repository Structure

```
miner_b/
├── phase0_testnet/           # Testnet validation phase
│   └── streetvision-subnet/  # Official NATIX subnet code (Git submodule)
├── streetvision_cascade/     # Production 6-stage cascade
│   ├── configs/              # Configuration files
│   ├── scripts/              # Utility scripts (model downloads, etc.)
│   ├── requirements.txt      # Python dependencies
│   └── [models/data ignored] # Large files managed externally
├── START_HERE.md             # Getting started guide
├── REALISTIC_DEPLOYMENT_PLAN.md  # Complete deployment strategy
├── COMPLETE_DEPLOYMENT_PLAN.md   # Phase-by-phase roadmap
└── README.md                 # This file
```

## Quick Start

### Prerequisites

- **GPU:** NVIDIA RTX 3090/4090 (24GB VRAM) or equivalent
- **Storage:** 200GB+ for models and data
- **Python:** 3.10+
- **CUDA:** 12.1+

### Installation

1. Clone repository:
```bash
git clone [YOUR_REPO_URL] miner_b
cd miner_b
```

2. Set up virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
cd streetvision_cascade
pip install -r requirements.txt
```

4. Download models (60GB):
```bash
python scripts/download_models.py
```

5. Download NATIX dataset:
```bash
# Contact NATIX team for official dataset access
# Expected: 8,000 labeled images (roadwork vs no-roadwork)
```

### Training Stage 1 (DINOv3)

See [REALISTIC_DEPLOYMENT_PLAN.md](REALISTIC_DEPLOYMENT_PLAN.md) for complete training workflow.

```bash
# Train DINOv3 classifier head (2-3 hours on RTX 3090)
python train_stage1_head.py

# Validate thresholds
python validate_stage1_thresholds.py

# Expected: 96-97% accuracy, 60% exit rate at threshold 0.88
```

## Model Architecture

**6-Stage Cascade:**

1. **Stage 1:** DINOv3-vith16plus (3.36GB) - Fast binary classifier (60% exit rate)
2. **Stage 2a:** RF-DETR-Medium - Object detection ensemble
3. **Stage 2b:** YOLOv12-X - Object detection ensemble
4. **Stage 3a:** GLM-4.6V-Flash (20GB) - Vision-language reasoning (images)
5. **Stage 3b:** Molmo-2-8B (33GB) - Vision-language reasoning (video)
6. **Stage 4:** Florence-2-Large (1.5GB) - OCR fallback

**Total:** 60.6GB models, 21GB loaded in VRAM (quantized)

## Documentation

- **[START_HERE.md](START_HERE.md)** - Quick overview
- **[REALISTIC_DEPLOYMENT_PLAN.md](REALISTIC_DEPLOYMENT_PLAN.md)** - Complete 5,036-line deployment guide
- **[COMPLETE_DEPLOYMENT_PLAN.md](COMPLETE_DEPLOYMENT_PLAN.md)** - Phase 0-5 roadmap
- **[COMPLETE_DEPLOYMENT_PLAN_PART2.md](COMPLETE_DEPLOYMENT_PLAN_PART2.md)** - Scaling guide (Weeks 2-12)

## Key Features

✅ **Daily Hard-Case Mining** - FiftyOne + SAM 3 automated workflow
✅ **vLLM-Omni** - Unified multimodal serving (zero model-swap latency)
✅ **TwelveLabs Marengo** - Long-video filtering (10× cost reduction)
✅ **M-GRPO** - Stable self-learning (Month 7+, zero manual annotation)
✅ **AutoAWQ** - 4-bit quantization for VLMs
✅ **TensorRT** - FP16/INT8 optimization for vision models

## Expected Performance

| Month | Accuracy | Ranking | Revenue |
|-------|----------|---------|---------|
| 1 | 96.5% | Top 30-40 | $150-300 |
| 2 | 97.2% | Top 20-25 | $350-600 |
| 3 | 97.8% | Top 15-20 | $600-1,000 |
| 6 | 98.2% | Top 10-15 | $900-1,500 |
| 12 | 98.6% | **Top 5-8** | **$1,200-2,000** |

## Cost Structure

**Hardware:**
- GPU rental: $150-200/month (RTX 3090 24GB)
- TAO registration: $280-350 one-time (0.7 TAO)

**Software:**
- 98% FREE (open-source: PyTorch, vLLM, FiftyOne, SAM 3, etc.)
- Optional: TwelveLabs ($0-20/month), Cosmos synthetics ($8-16/month)

**Total Year 1:** $4,100-4,300 costs → $6,800-35,000 revenue = **$2,500-30,700 profit**

## Competitive Advantages

1. **Daily Hard-Case Mining:** +0.2-0.5% accuracy/week (competitors train monthly)
2. **SAM 3 Auto-Annotation:** 6 seconds vs 400 minutes manual
3. **vLLM-Omni:** Zero model-swap latency (40% faster inference)
4. **M-GRPO (Month 7+):** Autonomous self-learning, no human labels

## License

[Specify your license - e.g., MIT, Apache 2.0, GPL, etc.]

## Contact

[Your contact information]

## Acknowledgments

- Bittensor Foundation
- NATIX Network
- Meta AI (DINOv3, SAM 3)
- Allen Institute (Molmo)
- vLLM team
