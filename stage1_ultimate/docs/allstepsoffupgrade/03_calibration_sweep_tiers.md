## Multi-Objective Calibration Ensemble (2025)

**Goal**: ECE 0.012-0.020 (vs 0.025-0.03 single method), -40% ECE.

**Config**: `configs/phase5/calibration_ultimate.yaml`
```yaml
phase5:
  calibration:
    mode: multi_objective_ensemble
    methods:
      tier1: [isotonic, temperature]
      tier2: [platt, beta]
      tier3: [dirichlet, spline, ensemble]
    fusion:
      learnable_weights: true
      optimize_on: val_calib
```

**Implementation**: Fit all methods on VAL_CALIB, compute ECE+MCC for each, select best with MCC-drop guardrail (max 0.02). Ensemble: learnable weighted combination of top 2-3.

**Selection**: Primary=ECE, secondary=MCC (reject if MCC drops >0.02).

**Outputs**: `calibration_summary.json` with winner + all metrics.
