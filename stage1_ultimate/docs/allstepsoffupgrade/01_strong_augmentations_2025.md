## Strong Augmentations (2025)

**Goal**: +14% precision, +50% robustness without hurting MCC.

**Config**: `configs/data/augmentation_2025.yaml`
```yaml
data:
  augmentation:
    tier: strong
    strong:
      trivial_augment_wide: {enabled: true, num_magnitude_bins: 31}
      augmix: {enabled: true, severity: 3, mixture_width: 3, alpha: 1.0}
      random_erasing: {enabled: true, p: 0.5}
    ablation:
      enabled: true
      mcc_drop_threshold: 0.03
```

**Implementation**: Update `src/data/natix_dataset.py:get_dinov3_transforms()` to read `cfg.data.augmentation.tier` and use `torchvision.transforms.v2`:
- `v2.TrivialAugmentWide()` (replaces RandAugment)
- `v2.AugMix()` (robustness)
- `v2.RandomErasing()` (occlusion)

**Ablation gate**: Run 1-3 epochs, compare MCC on VAL_SELECT. Reject if MCC drops >0.03.

**Split rule**: Augmentation selection on VAL_SELECT only. VAL_CALIB untouched.
