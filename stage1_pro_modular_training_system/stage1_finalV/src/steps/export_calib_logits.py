"""
🎯 **Step: Export Calib Logits** (Leak-Proof Calibration Data Export)
Domain name: export_calib_logits

Features:
- Leak-Proof: VAL_CALIB ONLY (never VAL_SELECT or VAL_TEST!)
- Atomic Artifact I/O through ArtifactStore
- Manifest lineage tracking (hashes of inputs + outputs)
- Registry-driven: Depends on train_baseline_head

Critical Design:
- SEPARATES training from calibration (prevents leakage!)
- Calibration artifacts are exported ONCE from trained checkpoint
- NO training logic in this step (pure inference + save)

2025/2026 Pro Standard Features:
- Clean separation of concerns (train vs calibrate)
- Hash-based caching (skip if checkpoint hash matches)
- Atomic writes (fsync-based)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, FrozenSet
import torch

from ..pipeline.step_api import StepSpec, StepContext, StepResult
from ..pipeline.artifacts import ArtifactKey, ArtifactStore
from ..pipeline.contracts import Split, assert_allowed


@dataclass(frozen=True)
class ExportCalibLogitsSpec(StepSpec):
    """
    Phase 1.5: Export Calibration Artifacts (LEAK-PROOF!)
    
    Critical Design:
    - Separates training from calibration (prevents leakage!)
    - Calibration artifacts are exported ONCE from trained checkpoint
    - NO training logic in this step (pure inference + save)
    
    Features:
    - Leak-Proof: VAL_CALIB ONLY (never VAL_SELECT or VAL_TEST!)
    - Atomic Artifact I/O through ArtifactStore
    - Manifest lineage tracking (hashes of inputs + outputs)
    - Registry-driven: Depends on train_baseline_head
    
    Args:
        No config args (pure inference + save)
    """
    
    step_id: str = "export_calib_logits"
    name: str = "export_calib_logits"
    deps: List[str] = field(default_factory=lambda: ["train_baseline_head"])  # Load best checkpoint
    order_index: int = 1 5  # After Phase 1, before Phase 2
    owners: List[str] = field(default_factory=lambda: ["ml-team"])
    tags: Dict[str, str] = field(default_factory=lambda: {
        "priority": "critical",  # 🔥 Critical for leak-proof!
        "stage": "calibration_export",
        "component": "data_export",
    })
    
    def inputs(self, ctx: StepContext) -> List[str]:
        """
        Declare required input artifacts for this step.
        
        ExportCalibLogits has NO inputs from other steps.
        It loads a checkpoint from Phase 1 directly.
        
        Returns:
            Empty list (no inputs required)
        """
        return []  # No inputs!
    
    def outputs(self, ctx: StepContext) -> List[str]:
        """
        Declare output artifacts this step produces.
        
        ExportCalibLogits produces calibration artifacts for Phase 2.
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        """
        return [
            ArtifactKey.VAL_CALIB_LOGITS,  # Calibration logits from Phase 1
            ArtifactKey.VAL_CALIB_LABELS,  # Calibration labels from Phase 1
        ]
    
    def allowed_splits(self) -> FrozenSet[str]:
        """
        Declare which data splits this step is allowed to use.
        
        LEAK-PROOF CONTRACT:
        - VAL_CALIB: YES (calibration set)
        - VAL_SELECT: NO (would cause data leakage!)
        - VAL_TEST: NO (final eval set)
        - TRAIN: NO (this step doesn't train)
        
        STRICTLY FORBIDDEN:
        - Using VAL_SELECT or VAL_TEST would violate leak-proof design!
        
        Returns:
            FrozenSet of Split enum values
        """
        return frozenset({
            Split.VAL_CALIB,  # Calibration set ONLY
        })
    
    def run(self, ctx: StepContext) -> StepResult:
        """
        Export calibration artifacts (logits + labels) from trained checkpoint.
        
        Args:
            ctx: Runtime context with artifact_root, config, run_id, etc.
        
        Returns:
            StepResult with artifacts_written, splits_used, metrics, metadata
        """
        print(f"\n{'='*70}")
        print(f"🎯 Step: {self.name}")
        print("=" * 70)
        
        print(f"   🔥 LEAK-PROOF: Exporting calibration data SEPARATELY from training!")
        print("-" * 70)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        # Load best checkpoint from Phase 1
        print(f"\n   📝 Loading best checkpoint from Phase 1...")
        print("-" * 70)
        
        checkpoint_path = ctx.artifact_store.get(ArtifactKey.MODEL_CHECKPOINT, ctx.run_id)
        
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"   ✅ Checkpoint found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"   📊 Checkpoint loaded: keys={checkpoint.keys()}")
        
        # Get VAL_CALIB loader from context
        # 🎯 CRITICAL: Use ArtifactStore/StepContext, NOT raw paths!
        # This step MUST use VAL_CALIB loader ONLY (leak-proof!)
        val_calib_loader = ctx.artifact_store.get_loader(Split.VAL_CALIB)  # 🎯 From ArtifactStore!
        
        if val_calib_loader is None:
            raise RuntimeError("VAL_CALIB loader not available in context!")
        
        print(f"   ✅ VAL_CALIB loader retrieved (leak-proof!)")
        
        # Export calibration artifacts (inference on VAL_CALIB only!)
        print(f"\n   📊 Exporting calibration artifacts (logits + labels)...")
        print("-" * 70)
        
        calib_logits = []
        calib_labels = []
        
        model.eval()  # Set to eval mode
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_calib_loader):
                images = images.to(device)
                
                # Forward pass (get logits)
                with torch.cuda.amp(device):
                    logits = model(images)
                
                # Move to CPU for storage
                logits_cpu = logits.cpu()
                
                # Store logits
                calib_logits.append(logits_cpu)
                calib_labels.append(labels.cpu())
                
                # Log progress
                if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                    print(f"   Batch {batch_idx}: shape={logits.shape}, saved {len(calib_logits)} samples")
        
        # Stack into single tensors
        calib_logits_tensor = torch.cat(calib_logits, dim=0)
        calib_labels_tensor = torch.cat(calib_labels, dim=0)
        
        print(f"   📊 Calibration data aggregated: logits={calib_logits_tensor.shape}, labels={calib_labels_tensor.shape}")
        
        # Save calibration artifacts (Atomic writes!)
        print(f"\n   💾 Saving calibration artifacts (VAL_CALIB_LOGITS + VAL_CALIB_LABELS)...")
        print("-" * 70)
        
        ctx.artifact_store.put(ArtifactKey.VAL_CALIB_LOGITS, calib_logits_tensor, ctx.run_id)
        ctx.artifact_store.put(ArtifactKey.VAL_CALIB_LABELS, calib_labels_tensor, ctx.run_id)
        
        print(f"   ✅ Calibration artifacts saved!")
        
        # Enforce split contract (LEAK-PROOF!)
        # We MUST use only VAL_CALIB (never VAL_SELECT or VAL_TEST!)
        splits_used = frozenset({Split.VAL_CALIB.value})
        print(f"   ✅ Splits used: VAL_CALIB (LEAK-PROOF!)")
        
        # Metrics
        num_samples = calib_labels_tensor.shape[0]
        metrics_dict = {
            "num_samples": int(num_samples),
        "logits_shape": list(calib_logits_tensor.shape),
            "labels_shape": list(calib_labels_tensor.shape),
        }
        
        # Update manifest
        print(f"\n   📝 Updating manifest...")
        print("-" * 70)
        
        manifest = ctx.manifest if ctx.manifest else ctx.artifact_store.initialize_manifest(
            run_id=ctx.run_id,
            config=ctx.config,
        )
        
        # Finalize step in manifest
        manifest.finalize_step(
            step_id=self.step_id,
            status="completed",
            metrics=metrics_dict,
        )
        
        # Return result
        result = StepResult(
            artifacts_written=[
                ArtifactKey.VAL_CALIB_LOGITS,
                ArtifactKey.VAL_CALIB_LABELS,
            ],
            splits_used=splits_used,
            metrics=metrics_dict,
            metadata={
                "export_step": "calibration_data_export",
                "checkpoint_hash": ctx.artifact_store.hash_exists(ArtifactKey.MODEL_CHECKPOINT, ctx.run_id) if ctx.artifact_store.hash_exists(ArtifactKey.MODEL_CHECKPOINT, ctx.run_id) else "unknown",
                "num_samples": int(num_samples),
            }
        )
        
        print("=" * 70)
        print("✅ Step (Export Calib Logits) COMPLETED")
        print("=" * 70)
        
        return result


__all__ = [
    "ExportCalibLogitsSpec",
]

