## Task PEFT (DoRA + RSLoRA + PiSSA)

**Goal**: +5-8% MCC vs standard LoRA, faster convergence.

**Config**: `configs/model/task_peft_dora.yaml`
```yaml
model:
  peft:
    enabled: true
    method: dora
    use_rslora: true
    init_lora_weights: pissa  # fallback to gaussian if unsupported
    rank: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj"]
```

**Implementation**: Apply to `self.net["backbone"].model` in `DINOv3Classifier`:
```python
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(
    r=cfg.model.peft.rank,
    lora_alpha=cfg.model.peft.alpha,
    use_rslora=cfg.model.peft.use_rslora,
    use_dora=(cfg.model.peft.method == "dora"),
    init_lora_weights=cfg.model.peft.init_lora_weights,
    target_modules=cfg.model.peft.target_modules,
    task_type=TaskType.SEQ_CLS,
)
peft_model = get_peft_model(backbone.model, lora_config)
```

**When**: After Phase-4 SimCLR stable. Train on TRAIN, validate on VAL_SELECT. Re-run Phase-2 after.
