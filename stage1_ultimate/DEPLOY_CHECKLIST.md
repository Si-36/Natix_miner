# Deployment Checklist - Copy-Paste Commands

## LOCAL PREPARATION (Run These Commands)

```bash
cd /home/sina/projects/miner_b/stage1_ultimate

# 1. Make scripts executable
chmod +x scripts/*.sh scripts/*.py

# 2. Verify eval gateway (ensures no MCC drift)
python3 scripts/verify_eval_gateway.py

# 3. Run smoke test (CRITICAL - must pass!)
bash scripts/smoke_test_local.sh
# Takes ~5 minutes
# If FAILS, do NOT go to SSH

# 4. Clean old outputs (saves space)
rm -rf outputs/stage1_ultimate/runs/*

# 5. Push to GitHub
git add .
git commit -m "Production-ready: smoke test passed"
git push
```

**STOP HERE if smoke test fails!**

---

## SSH GPU SERVER (After Renting)

```bash
# Get SSH details from vast.ai/runpod
# Example: ssh -p 12345 root@1.2.3.4

# ============================================
# ON SSH SERVER:
# ============================================

# 1. Clone repo
cd /workspace
git clone https://github.com/YOUR_USERNAME/natix-stage1-ultimate.git
cd natix-stage1-ultimate

# 2. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# 3. Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
# Should show: CUDA: True, GPUs: 2

# 4. Download dataset
python3 scripts/download_full_dataset.py
# OR upload with: scp -P PORT local_data.tar.gz root@HOST:/workspace/data/

# 5. Generate splits.json
python3 scripts/generate_splits.py

# 6. Update config
vim configs/config.yaml
# Change:
#   data_root: /workspace/data/natix_subset
#   splits_json: outputs/splits.json

# 7. Run FULL pipeline
python3 scripts/train_cli_v2.py \
    pipeline.phases=[phase1,phase2,phase4,phase5,phase6] \
    hardware.num_gpus=2

# Takes ~6-8 hours
# Monitor in another terminal: nvidia-smi -l 1

# 8. Download results (FROM LOCAL)
scp -P PORT -r root@HOST:/workspace/natix-stage1-ultimate/outputs/stage1_ultimate/runs/LATEST/ ./results/
```

---

## What Each Phase Does

| Phase | Input | Output | Time |
|-------|-------|--------|------|
| **1** | splits.json, data | model_best.pth, val_calib_logits.pt | 2-3h |
| **2** | val_calib_logits.pt | best_threshold.json, threshold_sweep.csv | 5-10min |
| **4** | phase1 checkpoint | explora_merged.pth | 3-4h |
| **5** | val_calib_logits.pt | calibration_params.json | 10min |
| **6** | all artifacts | bundle.json (deployment) | 2min |

---

## Troubleshooting

**Smoke test fails?**
```bash
# Check logs
tail -f outputs/stage1_ultimate/runs/*/logs/*.log

# Verify imports
python3 -c "from streetvision.eval import compute_mcc; print('OK')"
```

**SSH CUDA not found?**
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch
pip uninstall torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Pipeline crashes?**
```bash
# Just re-run same command (resume logic will skip completed phases)
python3 scripts/train_cli_v2.py pipeline.phases=[phase1,phase2,phase4,phase5,phase6] hardware.num_gpus=2
```

---

## Expected Results

After full pipeline:
- ✅ `outputs/stage1_ultimate/runs/YYYYMMDDTHHMMSS/` directory
- ✅ 5 phase directories (phase1, phase2, phase4, phase5, phase6)
- ✅ Each has `manifest.json`
- ✅ `phase6_bundle/export/bundle.json` (deployment artifact)
- ✅ Accuracy: ~77% (baseline ~69%, +8% from ExPLoRA)
- ✅ ECE reduction: 50-75%

---

**RULE: Run smoke test LOCAL before SSH. No exceptions!**
