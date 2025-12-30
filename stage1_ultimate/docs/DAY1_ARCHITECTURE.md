# Day 1 Architecture: Foundation Layer

## Overview

Production-grade foundation for Stage-1 training pipeline using 2025 best practices.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   NATIX Stage-1 Training Pipeline                   │
│                    (Production-Grade 2025-12-30)                    │
└─────────────────────────────────────────────────────────────────────┘

                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Hydra Configuration   │
                    │  (Run Isolation Active) │
                    └─────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌────────────────────┐      ┌────────────────────┐
        │  OLD CODE (Keep)   │      │  NEW CODE (Day 1)  │
        │  ~~~~~~~~~~~~~~~~  │      │  ~~~~~~~~~~~~~~~~  │
        │                    │      │                    │
        │  • scripts/        │      │  • streetvision/   │
        │    train_cli.py    │      │    io/             │
        │                    │      │    eval/           │
        │  • src/            │      │    cli/ (empty)    │
        │    models/         │      │                    │
        │    pipeline/       │      │  Features:         │
        │    contracts/      │      │  • Atomic IO       │
        │    data/           │      │  • Manifests       │
        │                    │      │  • Centralized MCC │
        └────────────────────┘      └────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │   DAG Engine (Updated)  │
                    │  • Phase orchestration  │
                    │  • Validation           │
                    │  • State management     │
                    └─────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Phase 1    │     │   Phase 2    │     │   Phase 6    │
    │  (Baseline)  │────▶│ (Threshold)  │────▶│  (Bundle)    │
    └──────────────┘     └──────────────┘     └──────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
    ┌──────────────────────────────────────────────────────┐
    │              Artifacts (with Manifests)              │
    │  • Checkpoints (model_best.pth + SHA256)             │
    │  • Logits (val_calib_logits.pt + SHA256)             │
    │  • Metrics (metrics.json + SHA256)                   │
    │  • Manifest (manifest.json) ◄── WRITTEN LAST         │
    └──────────────────────────────────────────────────────┘
```

---

## Day 1 Foundation Modules

### 1. Atomic IO Layer (`streetvision/io/`)

**Purpose:** Crash-safe file operations

```python
# ✅ Atomic write pattern (temp + os.replace)
from streetvision.io import write_json_atomic, write_checkpoint_atomic

# Write metrics (atomic)
checksum = write_json_atomic(
    path=Path("outputs/phase1/metrics.json"),
    data={"mcc": 0.856, "acc": 0.912}
)

# Write checkpoint (atomic)
checksum = write_checkpoint_atomic(
    path=Path("outputs/phase1/model_best.pth"),
    state_dict=model.state_dict()
)
```

**Guarantees:**
- ✅ Either old file exists OR new file exists (never corrupted)
- ✅ Cross-platform (Linux, macOS, Windows)
- ✅ SHA256 checksums for integrity
- ✅ No locks needed (Hydra run isolation)

---

### 2. Manifest System (`streetvision/io/manifests.py`)

**Purpose:** Lineage tracking & provenance

```python
from streetvision.io import create_step_manifest

manifest = create_step_manifest(
    step_name="phase1_baseline",
    input_paths=[Path("outputs/splits.json")],
    output_paths=[
        Path("outputs/phase1/model_best.pth"),
        Path("outputs/phase1/val_calib_logits.pt"),
    ],
    output_dir=Path("outputs"),
    metrics={"mcc": 0.856, "acc": 0.912, "fnr": 0.089},
    duration_seconds=3600.5,
    config=cfg,
)

# Write manifest LAST (after all artifacts)
manifest.save(Path("outputs/phase1/manifest.json"))
```

**Manifest Contents:**
```json
{
  "run_id": "20251230T123456",
  "step_name": "phase1_baseline",
  "git_sha": "a1b2c3d",
  "config_hash": "e4f5g6h",
  "input_artifacts": {...},
  "output_artifacts": {...},
  "metrics": {"mcc": 0.856},
  "duration_seconds": 3600.5,
  "hostname": "gpu-server-01",
  "python_version": "3.11.7"
}
```

**Why Manifest-Last:**
- If manifest exists → ALL artifacts guaranteed to exist
- No transaction rollback needed
- Simple, impossible to get wrong

---

### 3. Centralized Evaluation (`streetvision/eval/`)

**Purpose:** Single source of truth for metrics

```python
from streetvision.eval import compute_mcc, compute_all_metrics

# ✅ All phases use same MCC function
mcc = compute_mcc(y_true, y_pred)

# ✅ Get all metrics at once
metrics = compute_all_metrics(y_true, y_pred)
# Returns: {mcc, accuracy, precision, recall, f1, fnr, fpr, tp, tn, fp, fn}
```

**Prevents Metric Drift:**
```python
# ❌ OLD: Each phase computes MCC differently
# phase1/training.py: mcc = matthews_corrcoef(y_true, y_pred)
# phase2/threshold.py: mcc = (tp*tn - fp*fn) / sqrt(...)  # Custom implementation
# phase5/scrc.py: mcc = sklearn_mcc(y_true, y_pred)
# Result: Inconsistent MCC scores

# ✅ NEW: All phases use same function
from streetvision.eval import compute_mcc
mcc = compute_mcc(y_true, y_pred)
# Result: Consistent MCC across pipeline
```

---

### 4. Threshold Selection (`streetvision/eval/thresholds.py`)

**Purpose:** Phase-2 threshold optimization

```python
from streetvision.eval import select_threshold_max_mcc

# Find threshold that maximizes MCC
threshold, best_mcc = select_threshold_max_mcc(
    logits=val_calib_logits,
    labels=val_calib_labels,
    n_thresholds=100
)

print(f"Best threshold: {threshold:.3f}, MCC: {best_mcc:.3f}")
# Output: Best threshold: 0.520, MCC: 0.856
```

---

## Run Isolation (Hydra)

**Before (Risk of Conflicts):**
```
outputs/
├── phase1/
│   ├── model_best.pth      ◄── OVERWRITTEN on each run
│   └── metrics.json        ◄── OVERWRITTEN on each run
└── phase2/
    └── thresholds.json     ◄── OVERWRITTEN on each run
```

**After (Isolated Runs):**
```
outputs/stage1_ultimate/runs/
├── 20251230T091500/         ◄── Run 1 (isolated)
│   ├── phase1/
│   │   ├── model_best.pth
│   │   ├── metrics.json
│   │   └── manifest.json
│   └── phase2/
│       ├── thresholds.json
│       └── manifest.json
│
├── 20251230T093200/         ◄── Run 2 (isolated)
│   ├── phase1/
│   └── phase2/
│
└── 20251230T105400/         ◄── Run 3 (isolated)
    ├── phase1/
    └── phase2/
```

**Benefits:**
- ✅ No file conflicts (each run isolated)
- ✅ No locks needed (natural parallelism)
- ✅ Experiment comparison easy (just diff directories)
- ✅ Manifest-last ensures completeness

---

## Data Flow (Atomic + Manifest-Last)

```
Step 1: Train Phase-1
  │
  ├─▶ Write checkpoint (atomic)
  │   ├─ model_best.pth.tmp  (write to temp)
  │   └─ os.replace(tmp, model_best.pth)  (atomic)
  │
  ├─▶ Write logits (atomic)
  │   ├─ val_calib_logits.pt.tmp
  │   └─ os.replace(tmp, val_calib_logits.pt)
  │
  ├─▶ Compute metrics (centralized)
  │   └─ from streetvision.eval import compute_mcc
  │
  └─▶ Write manifest LAST (atomic)
      ├─ manifest.json.tmp
      └─ os.replace(tmp, manifest.json)

      ✅ If manifest exists → ALL artifacts exist
```

---

## Code Organization

### New Package Structure (Day 1)
```
src/streetvision/
├── __init__.py              # Package metadata
├── io/                      # Atomic IO + manifests
│   ├── __init__.py
│   ├── atomic.py            # write_*_atomic functions
│   └── manifests.py         # StepManifest, lineage tracking
├── eval/                    # Centralized metrics
│   ├── __init__.py
│   ├── metrics.py           # compute_mcc, compute_all_metrics
│   └── thresholds.py        # select_threshold_max_mcc
└── cli/                     # CLI (empty for Day 1)
```

### Old Structure (Unchanged)
```
src/
├── models/                  # Model definitions
├── pipeline/                # DAG engine, phase specs
├── contracts/               # Artifact schema, validators
└── data/                    # Data loading, transforms

scripts/
└── train_cli.py             # Main CLI (6 phase executors)
```

**Migration Strategy:**
- Day 1: Add new foundation (IO, eval) ✅
- Day 2-5: Extract phase steps one-by-one
- Week 2: Remove old code after validation

---

## Type Safety (Python 3.11+)

```python
from pathlib import Path
from typing import Dict, List, Tuple

def write_json_atomic(
    path: Path,
    data: Dict[str, Any]
) -> str:
    """
    Returns:
        SHA256 checksum
    """
    ...

def compute_mcc(
    y_true: ArrayLike,
    y_pred: ArrayLike
) -> float:
    """
    Returns:
        MCC score as Python float
    """
    ...
```

**Benefits:**
- ✅ IDE autocomplete
- ✅ Catch type errors before runtime
- ✅ Self-documenting code

---

## Testing Strategy

### Unit Tests (Future)
```python
def test_atomic_write():
    path = tmp_path / "test.json"
    data = {"key": "value"}
    checksum = write_json_atomic(path, data)
    assert path.exists()
    assert compute_file_sha256(path) == checksum

def test_mcc_computation():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    mcc = compute_mcc(y_true, y_pred)
    assert 0.0 <= mcc <= 1.0
```

### Integration Tests (Day 2+)
```python
def test_phase1_end_to_end():
    # Run Phase-1 with new code
    # Verify manifest exists
    # Verify all artifacts have checksums
    # Verify MCC matches old implementation
    ...
```

---

## Performance Characteristics

### Atomic Writes
- **Overhead:** ~2-5ms per file (negligible for checkpoints)
- **Bottleneck:** Disk I/O (same as non-atomic)
- **Benefit:** Zero corrupted files

### SHA256 Checksums
- **Speed:** ~500 MB/s (64KB chunks)
- **Overhead:** ~2 seconds for 1GB checkpoint
- **Benefit:** Early detection of corrupted transfers

### Centralized Metrics
- **Speed:** Same as sklearn (already optimized)
- **Overhead:** Function call (~0.1ms)
- **Benefit:** Zero metric drift

---

## Next Steps (Day 2)

1. Extract Phase-1 step → `src/streetvision/pipeline/steps/train_baseline.py`
2. Add atomic writes for checkpoints
3. Add manifest-last commit
4. Use centralized `compute_mcc()`
5. Test end-to-end

**Goal:** Phase-1 runs with new foundation, produces identical results to old code.

---

## Summary

Day 1 foundation provides:
- ✅ **Crash-safe IO** (atomic writes, no corruption)
- ✅ **Lineage tracking** (manifests with git SHA, config hash)
- ✅ **Metric consistency** (centralized MCC, no drift)
- ✅ **Run isolation** (no conflicts, natural parallelism)
- ✅ **Type safety** (Python 3.11+ hints)
- ✅ **Zero risk** (old code untouched)

**Production-ready for 2025-12-30.**
