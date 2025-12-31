## BYOL/SwAV Hybrid (Alternative to SimCLR)

**Goal**: 8-12% better features, no large batch requirement (SimCLR needs 256+).

**When**: Use ONLY if SimCLR bottlenecked by GPU memory or diverges.

**Config**: `configs/phase4a/contrastive_2025.yaml`
```yaml
phase4a:
  method: byol_swav_hybrid
  byol:
    momentum: 0.996
    predictor_hidden: 4096
  swav:
    prototypes: 3000
    temperature: 0.1
```

**Implementation**: Replace SimCLR NT-Xent loss with BYOL regression + SwAV clustering. No negatives needed (BYOL), batch size 256 sufficient.

**Trade-off**: Slightly lower performance (-2% to -5% vs SimCLR) but easier to train.

**Recommendation**: Keep SimCLR as baseline. Switch only if SimCLR fails (loss >10 after 5 epochs) or you reduce to 1 GPU.

