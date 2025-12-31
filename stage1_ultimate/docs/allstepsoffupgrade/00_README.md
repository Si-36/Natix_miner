## Optional High-Impact Upgrades (2025/2026)

Quick reference for cutting-edge upgrades. Full details in `MASTER_PLAN.md` Appendix H.

### Files
- `01_strong_augmentations_2025.md` - TrivialAugmentWide + AugMix + ablation gate
- `02_task_peft_dora_rslora_pissa.md` - DoRA + RSLoRA + PiSSA init
- `03_calibration_sweep_tiers.md` - Multi-objective calibration ensemble
- `04_pytorch26_compile.md` - PyTorch 2.6 optimizations
- `05_byol_swav_hybrid.md` - BYOL/SwAV alternative to SimCLR
- `06_flexattention_cvfm.md` - FlexAttention for trained CVFM (optional)

### Priority
- **Tier 0**: PyTorch 2.6 compile, DoRA+RSLoRA+PiSSA, Strong aug → +30% speed, +5-8% MCC
- **Tier 1**: Multi-objective calibration → -40% ECE
- **Tier 2**: BYOL/SwAV (if SimCLR fails), FlexAttention (if num_views >5)
