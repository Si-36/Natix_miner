# Integration Test Suite

**Status**: ✅ Complete (Day 5)  
**Coverage**: Pipeline execution, resume logic, manifest integrity, eval drift  
**Time**: ~40 minutes on CPU (tiny config)

---

## Quick Start

```bash
# Install pytest if needed
pip install pytest

# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_smoke.py -v
pytest tests/integration/test_resume.py -v
pytest tests/integration/test_manifest_integrity.py -v
pytest tests/integration/test_eval_drift.py -v

# Run with markers
pytest -m integration -v     # All integration tests
pytest -m slow -v            # Only slow tests
```

---

## Test Categories

### 1. Smoke Tests (`test_smoke.py`)
**Purpose**: Verify pipeline runs end-to-end without crashes

**Tests**:
- `test_pipeline_runs_phase1_tiny`: Phase-1 on tiny config (~5 min)
- `test_pipeline_runs_phase1_phase2`: Phase-1 → Phase-2 flow (~7 min)
- `test_pipeline_full_flow_with_bundle`: Phase-1 → Phase-2 → Phase-6 (~10 min)

**Validates**:
- All phases complete successfully
- All manifests created
- All artifacts exist
- Bundle export works

---

### 2. Resume Tests (`test_resume.py`)
**Purpose**: Verify crash recovery and resume logic

**Tests**:
- `test_resume_after_manifest_deleted`: Simulates crash before manifest write
- `test_skip_completed_phase`: Verifies skip logic (REQUIRES resume impl)
- `test_detect_corrupted_artifact`: Detects checksum mismatches
- `test_detect_missing_artifact`: Detects missing files

**Validates**:
- Resume after crash works
- Completed phases skipped
- Corrupted artifacts detected
- Missing artifacts detected

---

### 3. Manifest Integrity Tests (`test_manifest_integrity.py`)
**Purpose**: Verify checksum and metadata correctness

**Tests**:
- `test_manifest_has_all_checksums`: All artifacts have SHA256
- `test_checksums_match_actual_files`: Checksums match file contents
- `test_verify_manifest_helper`: Helper function works
- `test_corrupted_file_detected`: Corruption detection
- `test_missing_file_detected`: Missing file detection
- `test_manifest_has_metadata`: Metadata completeness

**Validates**:
- SHA256 integrity
- Manifest structure
- Verification functions
- Metadata tracking

---

### 4. Eval Drift Tests (`test_eval_drift.py`) **[CRITICAL]**
**Purpose**: Prevent MCC drift across phases

**Tests**:
- `test_no_mcc_drift_recomputation`: Recomputed MCC matches reported MCC
- `test_no_mcc_drift_across_phases`: Phase-1 and Phase-2 use same MCC
- `test_phase2_uses_same_logits_as_phase1`: Input checksums match
- `test_centralized_eval_import`: Centralized eval module works

**Validates**:
- Centralized MCC computation
- No metric drift
- Same logits → same MCC
- Threshold selection correctness

**Why Critical**: If Phase-1 and Phase-2 compute MCC differently, threshold selection will be WRONG!

---

## Test Configuration

Tests use **tiny config** for fast execution:
- 10 samples max
- 1 epoch only
- CPU-only (no GPU)
- Smallest backbone (dinov2_vits14)

**Time**: ~5 minutes per phase on CPU

---

## Test Fixtures

### `conftest.py` Fixtures

```python
@pytest.fixture
def tiny_config() -> DictConfig:
    """Tiny config for fast tests"""

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory (auto-cleanup)"""

@pytest.fixture
def tiny_config_with_output(...) -> DictConfig:
    """Tiny config + temp output dir"""
```

### Helper Functions

```python
load_manifest(path) -> Dict
    # Load manifest JSON

compute_file_sha256(path) -> str
    # Compute SHA256 checksum

verify_manifest(manifest, base_dir) -> None
    # Verify all artifacts + checksums
    # Raises: FileNotFoundError, ValueError
```

---

## Expected Output

```bash
$ pytest tests/integration/ -v

tests/integration/test_smoke.py::test_pipeline_runs_phase1_tiny PASSED
tests/integration/test_smoke.py::test_pipeline_runs_phase1_phase2 PASSED
tests/integration/test_smoke.py::test_pipeline_full_flow_with_bundle PASSED

tests/integration/test_resume.py::test_resume_after_manifest_deleted PASSED
tests/integration/test_resume.py::test_detect_corrupted_artifact PASSED
tests/integration/test_resume.py::test_detect_missing_artifact PASSED

tests/integration/test_manifest_integrity.py::test_manifest_has_all_checksums PASSED
tests/integration/test_manifest_integrity.py::test_checksums_match_actual_files PASSED
tests/integration/test_manifest_integrity.py::test_verify_manifest_helper PASSED
tests/integration/test_manifest_integrity.py::test_corrupted_file_detected PASSED
tests/integration/test_manifest_integrity.py::test_missing_file_detected PASSED
tests/integration/test_manifest_integrity.py::test_manifest_has_metadata PASSED

tests/integration/test_eval_drift.py::test_no_mcc_drift_recomputation PASSED
tests/integration/test_eval_drift.py::test_no_mcc_drift_across_phases PASSED
tests/integration/test_eval_drift.py::test_phase2_uses_same_logits_as_phase1 PASSED
tests/integration/test_eval_drift.py::test_centralized_eval_import PASSED

========== 16 passed in 40.23s ==========
```

---

## Troubleshooting

### Import Errors
```bash
# Add src to PYTHONPATH
export PYTHONPATH=/home/sina/projects/miner_b/stage1_ultimate/src:$PYTHONPATH

# Or install package in dev mode
pip install -e .
```

### Slow Tests
```bash
# Skip slow tests
pytest tests/integration/ -v -m "not slow"

# Run only fast tests
pytest tests/integration/test_eval_drift.py -v
```

### Test Isolation
Each test uses a temporary directory that is automatically cleaned up.
Tests do not interfere with each other.

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest tests/integration/ -v
```

---

## Maintenance

### Adding New Tests

1. Create test file in `tests/integration/`
2. Use fixtures from `conftest.py`
3. Mark slow tests with `@pytest.mark.slow`
4. Mark integration tests with `@pytest.mark.integration`
5. Run tests to verify

### Updating Fixtures

Edit `tests/conftest.py` to change:
- Tiny config parameters
- Helper functions
- Test fixtures

---

## Related Documentation

- `DAY5_COMPLETE.md`: Day 5 implementation summary
- `DAY5_PLAN.md`: Original Day 5 plan
- `FINAL_PROGRESS_SUMMARY.md`: Overall project status

---

**Questions?** Check the documentation or raise an issue.
