# GPU Fix Applied

## Issue Found
The ViT detector was loading the model on CPU instead of GPU, even though CUDA was available and configured.

## Warning Message
```
Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.
```

## Fix Applied
Modified `/home/sina/projects/miner_b/phase0_testnet/streetvision-subnet/base_miner/detectors/vit_detector.py`

**Before:**
```python
def load_model(self):
    self.model = pipeline(
        "image-classification",
        model=AutoModelForImageClassification.from_pretrained(self.hf_repo),
        feature_extractor=AutoImageProcessor.from_pretrained(self.hf_repo, use_fast=True),
    )
```

**After:**
```python
def load_model(self):
    self.model = pipeline(
        "image-classification",
        model=AutoModelForImageClassification.from_pretrained(self.hf_repo),
        feature_extractor=AutoImageProcessor.from_pretrained(self.hf_repo, use_fast=True),
        device=0 if self.device.type == "cuda" else -1,  # 0 for CUDA device 0, -1 for CPU
    )
```

## Next Steps
1. **Stop the current miner** (Ctrl+C)
2. **Restart the miner** to apply the fix:
   ```bash
   cd /home/sina/projects/miner_b/phase0_testnet/streetvision-subnet
   ./start_miner.sh
   ```
3. **Verify GPU usage** by checking:
   - No more CPU warning in logs
   - `nvidia-smi` shows GPU memory usage
   - Faster inference times (should be 10-20x faster on GPU)

## Expected Improvements
- **Latency**: CPU inference ~100-200ms → GPU inference ~10-20ms
- **VRAM Usage**: ~1-2GB for the ViT model
- **Throughput**: Can handle more concurrent queries

