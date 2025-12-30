# Files Created During Day 1-2 Refactor

**Total**: 16 files (8 code files, 4 documentation files, 4 package __init__ files)

---

## Code Files (8 files - ~66KB)

### Day 1 Foundation (4 files - ~28KB)

1. **src/streetvision/io/atomic.py** (~5.2KB)
   - `write_file_atomic()` - Atomic file write using temp + os.replace
   - `write_json_atomic()` - Atomic JSON serialization
   - `write_checkpoint_atomic()` - Atomic PyTorch checkpoint writes
   - `write_torch_artifact_atomic()` - Atomic tensor writes
   - `compute_file_sha256()` - File checksum computation
   - `get_file_size()` - File size metadata

2. **src/streetvision/io/manifests.py** (~8.7KB)
   - `ArtifactInfo` dataclass - File metadata with SHA256
   - `StepManifest` dataclass - Complete lineage tracking
   - `create_step_manifest()` - Factory function with auto-metadata
   - `get_git_sha()` - Git commit provenance
   - `get_config_hash()` - Reproducibility via config fingerprints
   - `create_artifact_info()` - Helper for artifact metadata

3. **src/streetvision/eval/metrics.py** (~8.4KB)
   - `compute_mcc()` - Matthews Correlation Coefficient
   - `compute_accuracy()` - Classification accuracy
   - `compute_precision()` - Precision metric
   - `compute_recall()` - Recall/Sensitivity/TPR
   - `compute_f1()` - F1 score
   - `compute_confusion()` - Confusion matrix (TP, TN, FP, FN)
   - `compute_fnr()` - False Negative Rate
   - `compute_fpr()` - False Positive Rate
   - `compute_all_metrics()` - One-shot all metrics
   - `_to_numpy()` - Helper for type conversion

4. **src/streetvision/eval/thresholds.py** (~6.1KB)
   - `select_threshold_max_mcc()` - Find threshold maximizing MCC
   - `sweep_thresholds_binary()` - Return full threshold curve
   - `plot_threshold_curve()` - Visualization (optional matplotlib)

### Day 2 Steps (4 files - ~38KB)

5. **src/streetvision/pipeline/steps/train_baseline.py** (~11.5KB)
   - `run_phase1_baseline()` - Phase-1 step executor
   - `compute_validation_metrics()` - Val metrics using centralized functions
   - Features:
     - Atomic checkpoint writes
     - Validation metrics on val_calib
     - Manifest-last commit
     - Duration tracking
     - ExPLoRA checkpoint loading
     - Config embedding

6. **src/streetvision/pipeline/steps/sweep_thresholds.py** (~8.2KB)
   - `run_phase2_threshold_sweep()` - Phase-2 step executor
   - Features:
     - Selective prediction sweep
     - Atomic JSON writes
     - Full sweep curve saved to CSV
     - MCC at best threshold
     - Manifest-last commit
     - Duration tracking

7. **src/streetvision/pipeline/steps/export_bundle.py** (~10.3KB)
   - `run_phase6_bundle_export()` - Phase-6 step executor
   - `load_bundle()` - Bundle loading with checksum verification
   - Features:
     - Relative paths (portable)
     - SHA256 checksums for all artifacts
     - Policy auto-detection (threshold XOR scrc)
     - Bundle verification
     - Manifest-last commit

8. **scripts/train_cli_v2.py** (~7.8KB)
   - `main()` - Hydra entry point
   - `register_phase_executors_v2()` - Register production-grade steps
   - `resolve_phases()` - Phase name resolution
   - Features:
     - Uses new step modules (not nested functions)
     - Clean executor registration
     - State saving on success/failure
     - Error handling

---

## Package Init Files (4 files - ~2KB)

9. **src/streetvision/__init__.py** (~0.3KB)
   - Package metadata
   - Version info

10. **src/streetvision/io/__init__.py** (~0.5KB)
    - Exports: write_file_atomic, write_json_atomic, write_checkpoint_atomic
    - Exports: ArtifactInfo, StepManifest, create_step_manifest

11. **src/streetvision/eval/__init__.py** (~0.7KB)
    - Exports: compute_mcc, compute_all_metrics
    - Exports: select_threshold_max_mcc, sweep_thresholds_binary

12. **src/streetvision/pipeline/__init__.py** (~0.2KB)
    - Pipeline package marker

13. **src/streetvision/pipeline/steps/__init__.py** (~0.7KB)
    - Exports: run_phase1_baseline, run_phase2_threshold_sweep
    - Exports: run_phase6_bundle_export, load_bundle

---

## Documentation Files (4 files - ~50KB)

14. **DAY1_FOUNDATION_COMPLETE.md** (~12KB)
    - Day 1 implementation details
    - Foundation layer architecture
    - Testing results
    - Benefits achieved

15. **DAY2_STEPS_COMPLETE.md** (~18KB)
    - Day 2 step extraction details
    - Code comparisons (old vs new)
    - Testing strategy
    - Migration status

16. **docs/DAY1_ARCHITECTURE.md** (~14KB)
    - Architecture diagrams
    - Data flow patterns
    - Pattern explanations
    - Next steps

17. **PROGRESS_SUMMARY.md** (~16KB)
    - Overall progress
    - Remaining work
    - Key decisions
    - Success metrics

---

## Modified Files (2 files)

18. **pyproject.toml** (modified)
    - Added matplotlib>=3.9.0 dependency
    - Updated isort known_first_party to include "streetvision"

19. **configs/config.yaml** (modified)
    - Updated Hydra run.dir to ISO 8601 format
    - Added documentation for run isolation

---

## File Tree

```
src/streetvision/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── atomic.py
│   └── manifests.py
├── eval/
│   ├── __init__.py
│   ├── metrics.py
│   └── thresholds.py
└── pipeline/
    ├── __init__.py
    └── steps/
        ├── __init__.py
        ├── train_baseline.py
        ├── sweep_thresholds.py
        └── export_bundle.py

scripts/
└── train_cli_v2.py

docs/
└── DAY1_ARCHITECTURE.md

./
├── DAY1_FOUNDATION_COMPLETE.md
├── DAY2_STEPS_COMPLETE.md
├── PROGRESS_SUMMARY.md
└── FILES_CREATED_DAY1_DAY2.md (this file)
```

---

## Summary

### New Code:
- **8 code files** (~66KB total)
- **4 package init files** (~2KB total)
- **Total executable code**: ~68KB

### Documentation:
- **4 documentation files** (~50KB total)

### Modified:
- **2 existing files** (pyproject.toml, configs/config.yaml)

### Grand Total:
- **18 files** created/modified
- **~118KB** of new content

---

## Verification Commands

```bash
# Count lines of Python code
find src/streetvision -name "*.py" -exec wc -l {} + | tail -1

# Check syntax
find src/streetvision -name "*.py" -exec python3 -m py_compile {} \;

# View structure
tree src/streetvision -L 3 --charset ascii -I '__pycache__'
```

---

## File Purposes

### Atomic IO (src/streetvision/io/):
- **Purpose**: Crash-safe file operations
- **Pattern**: Temp file + os.replace
- **Benefit**: No corrupted checkpoints

### Manifests (src/streetvision/io/):
- **Purpose**: Lineage tracking
- **Pattern**: Manifest-last commit
- **Benefit**: If manifest exists → all artifacts exist

### Centralized Metrics (src/streetvision/eval/):
- **Purpose**: Single source of truth for MCC
- **Pattern**: Centralized functions
- **Benefit**: No metric drift across phases

### Step Modules (src/streetvision/pipeline/steps/):
- **Purpose**: Production-grade phase implementations
- **Pattern**: Atomic writes + manifest-last + centralized metrics
- **Benefit**: Modular, testable, maintainable

### Production CLI (scripts/train_cli_v2.py):
- **Purpose**: Use new step modules
- **Pattern**: Clean executor registration
- **Benefit**: Zero risk (old code untouched)

### Documentation:
- **Purpose**: Knowledge transfer and progress tracking
- **Pattern**: Comprehensive explanations + examples
- **Benefit**: Easy onboarding for future developers

---

✅ All files created successfully and ready for Day 3 testing
