"""
Phase 6: Bundle Export Step (Production-Grade 2025-12-30)

Improvements over old implementation:
- ✅ Atomic JSON writes (crash-safe)
- ✅ Manifest-last commit (lineage tracking)
- ✅ Relative paths (portable across machines)
- ✅ SHA256 checksums for all bundled artifacts
- ✅ Duration tracking
- ✅ Type-safe with proper error handling

Contract:
- Inputs: phase1_checkpoint, splits.json, ONE policy (threshold XOR scrc)
- Outputs:
  - export/bundle.json (deployment manifest with RELATIVE paths)
  - export/manifest.json (lineage + checksums) ◄── LAST

Bundle Format:
{
  "model_checkpoint": "phase1/model_best.pth",  ◄── RELATIVE path
  "policy_type": "threshold",
  "policy_path": "phase2/thresholds.json",      ◄── RELATIVE path
  "splits_json": "splits.json",                 ◄── RELATIVE path
  "artifact_checksums": {...},                   ◄── SHA256 hashes
  ...
}
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Import old modules (gradual migration)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from contracts.artifact_schema import ArtifactSchema

# Import new foundation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from streetvision.io import (
    compute_file_sha256,
    create_step_manifest,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


def run_phase6_bundle_export(
    artifacts: ArtifactSchema,
    cfg: DictConfig,
) -> None:
    """
    Run Phase 6: Bundle Export with production-grade practices

    Args:
        artifacts: Artifact schema (all file paths)
        cfg: Hydra configuration

    Outputs:
        - bundle.json: Deployment manifest with relative paths (atomic write + SHA256)
        - manifest.json: Lineage tracking (git SHA, config hash, checksums) ◄── LAST

    Bundle Contents:
        - Model checkpoint (phase1/model_best.pth)
        - Policy file (phase2/thresholds.json OR phase5/scrcparams.json)
        - Data splits (splits.json)
        - Artifact checksums (SHA256 for integrity verification)
        - Metadata (backbone ID, num classes, created timestamp)

    Critical:
        - Uses RELATIVE paths (portable across machines)
        - Includes SHA256 checksums (verify integrity after transfer)
        - Atomic write (no corrupted bundle.json)
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("PHASE 6: Bundle Export (Production-Grade 2025-12-30)")
    logger.info("=" * 80)

    # Ensure export directory exists
    artifacts.export_dir.mkdir(parents=True, exist_ok=True)

    # Determine which policy to use (CRITICAL: exactly ONE)
    if artifacts.scrcparams_json.exists():
        policy_path = artifacts.scrcparams_json
        policy_type = "scrc"
    elif artifacts.thresholds_json.exists():
        policy_path = artifacts.thresholds_json
        policy_type = "threshold"
    else:
        raise FileNotFoundError(
            "No policy file found! Need thresholds.json or scrcparams.json. "
            "Run Phase 2 (threshold) or Phase 5 (SCRC) first."
        )

    logger.info(f"Policy: {policy_type} ({policy_path})")

    # Verify required inputs exist
    required_inputs = [
        artifacts.phase1_checkpoint,
        policy_path,
        artifacts.splits_json,
    ]

    for input_path in required_inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Required input not found: {input_path}")

    # Compute SHA256 checksums for all artifacts (integrity verification)
    logger.info("Computing artifact checksums...")
    artifact_checksums = {
        "model_checkpoint": compute_file_sha256(artifacts.phase1_checkpoint),
        "policy_file": compute_file_sha256(policy_path),
        "splits_json": compute_file_sha256(artifacts.splits_json),
    }

    logger.info(
        f"✅ Checksums computed: model={artifact_checksums['model_checkpoint'][:12]}..."
    )

    # Create bundle manifest with RELATIVE paths (portable)
    bundle_data = {
        # RELATIVE paths (not absolute)
        "model_checkpoint": str(
            artifacts.phase1_checkpoint.relative_to(artifacts.output_dir)
        ),
        "policy_type": policy_type,
        "policy_path": str(policy_path.relative_to(artifacts.output_dir)),
        "splits_json": str(artifacts.splits_json.relative_to(artifacts.output_dir)),
        # Artifact checksums (integrity verification)
        "artifact_checksums": artifact_checksums,
        # Model metadata
        "num_classes": cfg.model.num_classes,
        "backbone_id": cfg.model.backbone_id,
        "freeze_backbone": cfg.model.freeze_backbone,
        "head_type": cfg.model.head_type,
        # Deployment metadata
        "created_at": datetime.now().isoformat(),
        "python_version": cfg.get("python_version", "3.11+"),
        # CRITICAL: Output dir reference (for resolving relative paths)
        "output_dir_info": {
            "note": "All paths are relative to output_dir",
            "example": "model_checkpoint = output_dir / 'phase1/model_best.pth'",
        },
    }

    # Write bundle.json (ATOMIC WRITE)
    bundle_checksum = write_json_atomic(artifacts.bundle_json, bundle_data)
    logger.info(
        f"✅ Bundle saved: {artifacts.bundle_json} (SHA256: {bundle_checksum[:12]}...)"
    )

    # Create and save MANIFEST (LAST STEP)
    duration_seconds = time.time() - start_time
    logger.info("Creating manifest (lineage tracking)...")

    manifest = create_step_manifest(
        step_name="phase6_bundle_export",
        input_paths=required_inputs,
        output_paths=[artifacts.bundle_json],
        output_dir=artifacts.output_dir,
        metrics={
            "policy_type": policy_type,
            "num_artifacts": len(artifact_checksums),
        },
        duration_seconds=duration_seconds,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    manifest_checksum = manifest.save(artifacts.export_dir / "manifest.json")
    logger.info(
        f"✅ Manifest saved: {artifacts.export_dir / 'manifest.json'} "
        f"(SHA256: {manifest_checksum[:12]}...)"
    )

    # Summary
    logger.info("=" * 80)
    logger.info("✅ Phase 6 Complete (Production-Grade)")
    logger.info(f"Duration: {duration_seconds:.1f} seconds")
    logger.info(f"Model:  {artifacts.phase1_checkpoint}")
    logger.info(f"Policy: {policy_path} ({policy_type})")
    logger.info(f"Bundle: {artifacts.bundle_json}")
    logger.info(f"Manifest: {artifacts.export_dir / 'manifest.json'}")
    logger.info("")
    logger.info("📦 Bundle Contents (Relative Paths):")
    logger.info(f"  - model: {bundle_data['model_checkpoint']}")
    logger.info(f"  - policy: {bundle_data['policy_path']}")
    logger.info(f"  - splits: {bundle_data['splits_json']}")
    logger.info("")
    logger.info("🔐 Artifact Checksums (SHA256):")
    for name, checksum in artifact_checksums.items():
        logger.info(f"  - {name}: {checksum[:16]}...")
    logger.info("=" * 80)


def load_bundle(bundle_path: Path, output_dir: Path) -> Dict:
    """
    Load a deployment bundle and resolve relative paths

    Args:
        bundle_path: Path to bundle.json
        output_dir: Output directory root (for resolving relative paths)

    Returns:
        Dict with resolved absolute paths

    Example:
        >>> bundle = load_bundle(Path("bundle.json"), Path("outputs"))
        >>> model_path = bundle["model_checkpoint_abs"]
        >>> torch.load(model_path)

    Verification:
        Computes SHA256 checksums and compares with bundle manifest
        to ensure integrity after transfer.
    """
    with open(bundle_path, "r") as f:
        bundle = json.load(f)

    # Resolve relative paths to absolute
    bundle["model_checkpoint_abs"] = output_dir / bundle["model_checkpoint"]
    bundle["policy_path_abs"] = output_dir / bundle["policy_path"]
    bundle["splits_json_abs"] = output_dir / bundle["splits_json"]

    # Verify checksums (integrity check)
    logger.info("Verifying artifact checksums...")
    for artifact_name, expected_checksum in bundle["artifact_checksums"].items():
        if artifact_name == "model_checkpoint":
            actual_checksum = compute_file_sha256(bundle["model_checkpoint_abs"])
        elif artifact_name == "policy_file":
            actual_checksum = compute_file_sha256(bundle["policy_path_abs"])
        elif artifact_name == "splits_json":
            actual_checksum = compute_file_sha256(bundle["splits_json_abs"])
        else:
            continue

        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Checksum mismatch for {artifact_name}! "
                f"Expected: {expected_checksum[:16]}..., "
                f"Got: {actual_checksum[:16]}..."
            )

    logger.info("✅ All checksums verified")

    return bundle
