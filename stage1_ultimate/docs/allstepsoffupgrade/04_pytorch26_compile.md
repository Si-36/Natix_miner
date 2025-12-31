## PyTorch 2.6 Compile Optimizations

**Goal**: 2× faster inference, 1.5× faster training vs PyTorch 2.5 baseline.

**Config**: `configs/training/pytorch26.yaml`
```yaml
hardware:
  pytorch_version: "2.6+"
  compiler:
    enabled: true
    stance: performance  # NEW: torch.compiler.set_stance()
    mode: max-autotune
    fullgraph: true
    dynamic: false
```

**Implementation**: Update training scripts:
```python
import torch
torch.compiler.set_stance("performance")  # NEW in 2.6
model = torch.compile(
    model,
    mode="max-autotune",  # Better than reduce-overhead
    fullgraph=True,       # Better optimization
    dynamic=False         # Static shapes = faster
)
```

**Validation**: Check MCC doesn't drop >2% after switching modes. Revert to `reduce-overhead` if unstable.

**When**: After Phase-2 MCC baseline stable. Use for production inference (amortize compile cost).

