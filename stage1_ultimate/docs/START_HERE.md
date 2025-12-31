## Stage1 Ultimate — START HERE
Updated: 2025-12-31 (Final Version)

### 🚀 QUICK START: Use the TODO Checklist!

**For step-by-step implementation, use: [`docs/TODO_CHECKLIST.md`](./TODO_CHECKLIST.md)**

This checklist contains **203 atomic tasks** organized day-by-day:
- ✅ Exact file paths and code changes
- ✅ Verification steps for each task
- ✅ Agent-friendly format
- ✅ Progress tracking

**Start with Day 0 (Tasks #1-8), then work through each day sequentially.**

---

### What you are doing
You will upgrade the training pipeline to maximize **MCC** (and keep **FNR** low) with:
- **Phase‑4 domain adaptation** (true unsupervised SimCLR)
- **Phase‑1 task training** (pro training knobs: BF16, grad accumulation, torch.compile, focal loss, strong aug)
- **Phase‑2 MCC sweep** with **5000 thresholds**
- **Phase‑5 calibration** (SCRC policy)
- **Phase‑6 export** (validator‑compatible bundle)
- Optional: **CVFM fusion** (inference + trained) + evaluation step

### Non‑negotiable data split rules (no leakage)
- **TRAIN**: training only (Phase‑4, Phase‑1, CVFM training)
- **VAL_SELECT**: early stopping/model selection + CVFM validation
- **VAL_CALIB**: Phase‑2 threshold fitting + Phase‑5 calibration fitting (no gradients)
- **VAL_TEST**: final report only

### One‑time setup (local or SSH)
```bash
cd /home/sina/projects/miner_b/stage1_ultimate
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### Data + splits (must exist before training)
```bash
python scripts/download_full_dataset.py --output-dir /workspace/data/natix_subset
python scripts/generate_splits.py --data-root /workspace/data/natix_subset
ls -lah outputs/splits.json
```

### Run the pipeline (single command)
```bash
python scripts/train_cli_v2.py \
  pipeline.phases=[phase4,phase1,phase2,phase5,phase6] \
  phase2.n_thresholds=5000 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  training.gradient_accumulation_steps=2 \
  hardware.compile=false \
  hardware.num_gpus=2
```

### Important note (so you don’t get blocked)
The command above assumes you have already applied the upgrade changes described in `docs/MASTER_PLAN.md`
(especially the Phase‑2 MCC sweep wiring and config keys like `phase2.n_thresholds`).

If you are running the repo **before** applying the upgrades, start with:

```bash
python scripts/train_cli_v2.py pipeline.phases=[phase1]
```

### What you should see in outputs
- `phase1/val_calib_logits.pt` + `phase1/val_calib_labels.pt`
- `phase2/thresholds.json` + `phase2/threshold_sweep.csv`
- `phase5_scrc/scrcparams.json`
- `export/bundle.json`
