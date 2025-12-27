---
name: Stage-1 Pro Modular Training System (Schema-Clean, Artifact-First)
overview: "Refactor the phased plan to remove duplicates, fix schema contradictions, unify training scripts, create separate policy artifacts (thresholds.json, gateparams.json, scrcparams.json), add JSON schemas, and enforce mutual exclusivity. Production exit policy: SCRC (Phase 6) becomes final default, softmax/gate are logged baselines only."
todos:
  - id: config_rename_exit_threshold
    content: "Rename exit_threshold → legacy_exit_threshold_for_logging in Stage1ProConfig. Add exit_policy: str = 'softmax' field. Remove any target_coverage references. Add target_fnr_exit: float = 0.02 as single constraint."
    status: pending
  - id: config_validate_exit_policy
    content: "Add validation: exit_policy must match phase capabilities (Phase 1: only 'softmax', Phase 3+: 'softmax' or 'gate', Phase 6: all three). Validate target_fnr_exit > 0 and <= 0.1."
    status: pending
    dependencies:
      - config_rename_exit_threshold
  - id: schemas_create_all
    content: "Create stage1_pro/schemas/ directory with JSON schemas: config.schema.json, splits.schema.json, thresholds.schema.json, gateparams.schema.json, scrcparams.schema.json, bundle.schema.json"
    status: pending
  - id: schemas_validation
    content: Implement schema validation on all artifact writes and loads using jsonschema library. Fail fast with clear error messages.
    status: pending
    dependencies:
      - schemas_create_all
  - id: script_unify_training
    content: DELETE references to 20_train_baseline.py, 21_train_gate.py, 22_train_riskaware.py. CREATE single 20_train.py with --phase {1..6} and --exit_policy {softmax,gate,scrc} arguments. Implement phase-based routing and validation.
    status: pending
  - id: script_separate_policy_artifacts
    content: Update 25_threshold_sweep.py to write thresholds.json (Phase 1). Update 33_calibrate_gate.py to write gateparams.json (NOT thresholds.json). Update 36_calibrate_scrc.py to write scrcparams.json. Each validates against schema.
    status: pending
  - id: script_bundle_manifest
    content: Update 50_export_bundle.py to create bundle.json manifest with active_exit_policy and file pointers. Enforce exactly ONE policy file in bundle. Validate all JSON files against schemas.
    status: pending
    dependencies:
      - schemas_validation
  - id: loss_remove_target_coverage
    content: Remove target_coverage parameter from SelectiveLoss. Change objective to minimize selective risk subject to FNR ≤ target_fnr_exit. Coverage maximized implicitly, not targeted explicitly.
    status: pending
  - id: inference_bundle_contract
    content: "Update inference bundle loader: Load bundle.json first to determine active_exit_policy. Load corresponding policy file. Validate against schema. NEVER read config.legacy_exit_threshold_for_logging for inference."
    status: pending
    dependencies:
      - script_bundle_manifest
  - id: cli_remove_target_coverage
    content: Remove --target_coverage from CLI entirely. Add --exit_policy argument with validation. Update help text to reflect single constraint (target_fnr_exit) and policy selection.
    status: pending
---

# Stage-1 Pro Modular Training System (Sc

hema-Clean, Artifact-First)

## Critical Fixes Applied

### 1. Removed Duplicates

**Unified Training Scripts:**

- **DELETE:** `20_train_baseline.py`, `21_train_gate.py`, `22_train_riskaware.py`
- **CREATE:** Single `20_train.py` with `--phase {1..6}` and `--exit_policy {softmax,gate,scrc}`
- Validates phase supports selected policy (e.g., Phase 1 only supports `softmax`)

**Separate Policy Artifacts (Mutually Exclusive):**

- `thresholds.json` - Phase 1 softmax policy only
- `gateparams.json` - Phase 3 gate policy only (NOT written into thresholds.json)
- `scrcparams.json` - Phase 6 SCRC policy only
- Bundle contains exactly ONE policy artifact, never multiple

**Removed Parameter Contradictions:**

- **DELETE:** All `--target_coverage` references from CLI, config, and SelectiveLoss
- **RENAME:** `exit_threshold` → `legacy_exit_threshold_for_logging` in config (monitoring only)
- **ADD:** `exit_policy: str = "softmax"` field in config (enforced mutually exclusive)

### 2. Schema Structure (4 Schemas Total)

**1. config.json** (Training-time settings)

```json
{
  "model_path": "...",
  "target_fnr_exit": 0.02,  // Single constraint
  "exit_policy": "softmax",  // "softmax" | "gate" | "scrc"
  "legacy_exit_threshold_for_logging": 0.88,  // Monitoring only
  "phase": 1,
  // ... all other training hyperparameters
}
```

**MUST NOT contain:** Chosen thresholds, split indices, calibration parameters**2. splits.json** (Data split artifact)

```json
{
  "val_select_indices": [...],
  "val_calib_indices": [...],
  "seed": 42,
  "class_balance": {...},
  "metadata": {...}
}
```

**3. thresholds.json** (Phase 1/2 production policy)

```json
{
  "exit_threshold": 0.88,
  "fnr_on_exits": 0.019,
  "coverage": 0.75,
  "exit_accuracy": 0.981,
  "sweep_results": [...],
  "val_calib_metrics": {...}
}
```

**4. gateparams.json** (Phase 3 production policy)

```json
{
  "gate_threshold": 0.90,
  "calibrator_type": "platt",
  "calibrator_params": {...},
  "fnr_on_exits": 0.018,
  "coverage": 0.78,
  "val_calib_metrics": {...}
}
```

**5. scrcparams.json** (Phase 6 production policy)

```json
{
  "lambda1": 0.85,
  "lambda2": 0.92,
  "scrc_variant": "SCRC-I",  // or "SCRC-T"
  "fnr_bound": 0.02,
  "coverage": 0.80,
  "val_calib_metrics": {...},
  "model_version": "...",
  "splits_version": "..."
}
```

**6. bundle.json** (Deployment manifest)

```json
{
  "active_exit_policy": "scrc",
  "policy_file": "scrcparams.json",
  "model_file": "model_best.pth",
  "splits_file": "splits.json",
  "config_file": "config.json",
  "metrics_file": "metrics.csv",
  "version": "1.0.0",
  "created_at": "..."
}
```



### 3. JSON Schema Validation

**Create `stage1_pro/schemas/` directory:**

- `config.schema.json` - Validates config.json structure
- `splits.schema.json` - Validates splits.json structure
- `thresholds.schema.json` - Validates thresholds.json structure
- `gateparams.schema.json` - Validates gateparams.json structure
- `scrcparams.schema.json` - Validates scrcparams.json structure
- `bundle.schema.json` - Validates bundle.json structure

**Validation Rules:**

- Validate on write (fail fast if invalid)
- Validate on load (fail fast if corrupted)
- Use `jsonschema` library for validation

### 4. Fixed SelectiveLoss (No target_coverage)

**Before (WRONG):**

```python
coverage_penalty = lambda_cov * max(0, target_coverage - realized_coverage)^2
```

**After (CORRECT):**

```python
# Phase 3/6: Control FNR ≤ target_fnr_exit, maximize coverage
# No coverage target - just maximize subject to FNR constraint
selective_risk = compute_error_on_accepted(gate_prob >= threshold)
# Coverage is maximized implicitly by minimizing selective_risk
```



### 5. Unified Training Script

**`stage1_pro/scripts/20_train.py`:**

```python
def main():
    parser.add_argument("--phase", type=int, choices=[1,2,3,4,5,6], required=True)
    parser.add_argument("--exit_policy", type=str, choices=["softmax","gate","scrc"], default="softmax")
    
    # Validate phase supports exit_policy
    if args.phase == 1 and args.exit_policy != "softmax":
        raise ValueError("Phase 1 only supports exit_policy='softmax'")
    if args.phase < 3 and args.exit_policy == "gate":
        raise ValueError("Gate policy requires Phase 3+")
    if args.phase < 6 and args.exit_policy == "scrc":
        raise ValueError("SCRC policy requires Phase 6")
    
    # Route to appropriate training mode
    if args.phase == 1:
        train_baseline(config)  # Single-head, CrossEntropyLoss
    elif args.phase == 3:
        train_with_gate(config)  # 3-head, SelectiveLoss
    elif args.phase == 6:
        train_with_conformal_risk(config)  # 3-head, ConformalRiskLoss
```



### 6. Bundle Export Rules

**`stage1_pro/scripts/50_export_bundle.py`:**

```python
def export_bundle(config, run_id):
    # Required files
    required = ["config.json", "splits.json", "model_best.pth", "metrics.csv"]
    
    # Exactly ONE policy file
    policy_files = {
        "softmax": "thresholds.json",
        "gate": "gateparams.json",
        "scrc": "scrcparams.json"
    }
    
    policy_file = policy_files[config.exit_policy]
    assert os.path.exists(policy_file), f"Policy file {policy_file} missing"
    
    # Create bundle.json manifest
    bundle_manifest = {
        "active_exit_policy": config.exit_policy,
        "policy_file": policy_file,
        "model_file": "model_best.pth",
        # ... all file pointers
    }
    
    # Validate: exactly one policy file exists
    for other_policy in ["thresholds.json", "gateparams.json", "scrcparams.json"]:
        if other_policy != policy_file and os.path.exists(other_policy):
            raise ValueError(f"Multiple policy files found! Only {policy_file} should exist.")
    
    # Copy all files to bundle directory
    # Validate all JSON files against schemas
```



### 7. Inference Contract

**Inference MUST:**

1. Load `bundle.json` to determine `active_exit_policy`
2. Load the corresponding policy file (`thresholds.json` OR `gateparams.json` OR `scrcparams.json`)
3. NEVER read `config.exit_threshold` or `config.legacy_exit_threshold_for_logging`
4. NEVER hard-code thresholds

**Example inference code:**

```python
def load_exit_policy(bundle_dir):
    with open(f"{bundle_dir}/bundle.json") as f:
        bundle = json.load(f)
    
    policy_file = bundle["policy_file"]
    exit_policy = bundle["active_exit_policy"]
    
    with open(f"{bundle_dir}/{policy_file}") as f:
        policy = json.load(f)
    
    if exit_policy == "softmax":
        return SoftmaxExitPolicy(policy["exit_threshold"])
    elif exit_policy == "gate":
        return GateExitPolicy(policy["gate_threshold"], policy["calibrator_params"])
    elif exit_policy == "scrc":
        return SCRCExitPolicy(policy["lambda1"], policy["lambda2"])
```



## Updated Implementation Tasks

### Config Changes

**Update `config_dataclass` todo:**

- Rename `exit_threshold` → `legacy_exit_threshold_for_logging: float = 0.88`
- Add `exit_policy: str = "softmax"` (validated: must match phase capabilities)
- Add `target_fnr_exit: float = 0.02` (single constraint)
- **REMOVE** any `target_coverage` references

**Update `config_validation` todo:**

- Validate `exit_policy` matches phase (Phase 1: only "softmax", Phase 3+: "softmax" or "gate", Phase 6: all three)
- Validate `target_fnr_exit` > 0 and <= 0.1
- Ensure `legacy_exit_threshold_for_logging` is only used for metrics, not inference

### Script Changes

**Update `script_train_full` todo:**

- **DELETE:** References to `20_train_baseline.py`, `21_train_gate.py`, `22_train_riskaware.py`
- **CREATE:** Single `20_train.py` with `--phase` and `--exit_policy` arguments
- Implement phase-based routing logic
- Save `exit_policy` to config.json

**Update `script_calibrate` todo:**

- `25_threshold_sweep.py` writes `thresholds.json` (Phase 1 only)
- `33_calibrate_gate.py` writes `gateparams.json` (Phase 3 only, NOT thresholds.json)
- `36_calibrate_scrc.py` writes `scrcparams.json` (Phase 6 only)
- Each script validates its output against corresponding JSON schema

**Update `script_export` todo:**

- Create `bundle.json` manifest with `active_exit_policy` and file pointers
- Enforce exactly ONE policy file in bundle
- Validate all JSON files against schemas before export
- Create bundle README documenting inference contract

### Loss Changes

**Update `loss_coverage_penalty` todo:**

- **REMOVE:** `target_coverage` parameter
- **CHANGE:** SelectiveLoss objective to "minimize selective risk subject to FNR ≤ target_fnr_exit"
- Coverage is maximized implicitly, not targeted explicitly

### Schema Tasks (NEW)

**Add `schemas_config` todo:**

- Create `stage1_pro/schemas/config.schema.json`
- Define structure: training hyperparameters, `target_fnr_exit`, `exit_policy`, `legacy_exit_threshold_for_logging`
- Validate on config.save() and config.load()

**Add `schemas_splits` todo:**

- Create `stage1_pro/schemas/splits.schema.json`
- Define structure: indices, seed, metadata

**Add `schemas_thresholds` todo:**

- Create `stage1_pro/schemas/thresholds.schema.json`
- Define structure: exit_threshold, metrics, sweep_results

**Add `schemas_gateparams` todo:**

- Create `stage1_pro/schemas/gateparams.schema.json`
- Define structure: gate_threshold, calibrator_type, calibrator_params, metrics

**Add `schemas_scrcparams` todo:**

- Create `stage1_pro/schemas/scrcparams.schema.json`
- Define structure: lambda1, lambda2, scrc_variant, bounds, version info

**Add `schemas_bundle` todo:**

- Create `stage1_pro/schemas/bundle.schema.json`
- Define structure: active_exit_policy, file pointers, version

**Add `validation_schemas` todo:**

- Implement schema validation on all artifact writes
- Implement schema validation on all artifact loads
- Use `jsonschema` library
- Fail fast with clear error messages

### Inference Changes

**Update `inference_bundle_loader` todo:**

- Load `bundle.json` first to determine `active_exit_policy`
- Load corresponding policy file based on `active_exit_policy`
- Validate policy file against schema
- **NEVER** read `config.legacy_exit_threshold_for_logging` for inference
- **NEVER** hard-code thresholds

## Phase-Specific Policy Evolution

**Phase 1-2:** `exit_policy="softmax"`, bundle contains `thresholds.json`**Phase 3-5:** `exit_policy="gate"` (optional, can still use "softmax"), bundle contains `gateparams.json` OR `thresholds.json`**Phase 6:** `exit_policy="scrc"` (production default), bundle contains `scrcparams.json`Previous policies remain as logged metrics only, never active in production.

## Research Item Verification

**SCRC:** Use arXiv 2512.12844 (SCRC-I default, SCRC-T optional)**Conformal Risk Training:** Use arXiv 2510.08748 + code repo reference**ExPLoRA:** Use arXiv 2406.10973 + official repo**F-SAM:** Use CVPR 2024 paper + implementation reference**DoRAN:** Mark as "DoRA (verified) + optional DoRAN (if implementation exists)"

## Key Principles (Updated)

1. **One Config Schema:** Training settings only, no policy parameters
2. **Separate Policy Artifacts:** thresholds.json, gateparams.json, scrcparams.json (mutually exclusive)
3. **Bundle Manifest:** bundle.json declares active policy and file pointers
4. **Schema Validation:** All artifacts validated on write/load
5. **Single Constraint:** `target_fnr_exit` only, maximize coverage implicitly
6. **Unified Training:** One `20_train.py` script with phase-based routing
7. **Inference Contract:** Load policy from bundle.json → policy file, never from config