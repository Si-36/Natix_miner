#!/usr/bin/env python3
"""
Tier 0 CLI - Training Pipeline Entry Point
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "src"))

from pipeline.artifacts import ArtifactStore, ArtifactKey
from pipeline.contracts import Split, assert_allowed
from pipeline.step_api import StepContext, StepSpec
from pipeline.manifest import RunManifest
from pipeline.registry import StepRegistry, resolve_execution_order

from steps.train_baseline_head import TrainBaselineHeadSpec
from steps.export_calib_logits import ExportCalibLogitsSpec
from steps.sweep_thresholds import SweepThresholdsSpec

_step_registry = StepRegistry()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tier 0 Training Pipeline - Modular DAG-based training system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target_step",
        type=str,
        default="sweep_thresholds",
        choices=[
            "train_baseline_head",
            "train_explora_head",
            "train_explora_ddp",
            "export_calib_logits",
            "sweep_thresholds",
        ],
        help="Target step to run (default: sweep_thresholds, runs full pipeline to target)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1, smoke test)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic/mock data instead of real dataset",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Override run ID (default: auto-generated)",
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        default="runs",
        help="Root directory for artifacts (default: runs)",
    )

    return parser.parse_args()


def generate_run_id():
    now = datetime.now()
    return now.strftime("%Y%m%dT%H%M%S")


def run_pipeline(args):
    print("=" * 70)
    print("Tier 0 Training Pipeline")
    print("=" * 70)
    print()

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = generate_run_id()

    print(f"Run ID: {run_id}")
    print()

    config = {
        "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "hidden_dim": 384,
        "num_classes": 2,
        "dropout": 0.1,
        "freeze_backbone": True,
        "training_max_epochs": args.epochs,
        "training_batch_size": args.batch_size,
        "training_learning_rate": args.learning_rate,
        "data_synthetic": args.synthetic,
    }

    print("\nFinal config:")
    print(f"   Model ID: {config['model_id']}")
    print(f"   Hidden dim: {config['hidden_dim']}")
    print(f"   Num classes: {config['num_classes']}")
    print(f"   Max epochs: {config['training_max_epochs']}")
    print(f"   Batch size: {config['training_batch_size']}")
    print(f"   Learning rate: {config['training_learning_rate']}")
    print(f"   Synthetic data: {config['data_synthetic']}")
    print()

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    artifact_store = ArtifactStore(artifact_root)
    print(f"Artifact root: {artifact_root}")
    print()

    manifest = RunManifest(run_id=run_id, resolved_config=config)
    print(f"Manifest initialized: {run_id}")
    print()

    target_step = args.target_step
    print(f"Target step: {target_step}")
    print()

    try:
        execution_order = _step_registry.resolve_execution_order(target_step)
    except ValueError as e:
        print(f"DAG resolution failed: {e}")
        sys.exit(1)

    print(f"Execution order: {' -> '.join(execution_order)}")
    print()

    for step_name in execution_order:
        print("=" * 70)
        print(f"Running: {step_name}")
        print("-" * 70)

        step_spec_class = _step_registry._step_specs[step_name]
        ctx = StepContext(
            step_id=step_name,
            config=config,
            run_id=run_id,
            artifact_root=artifact_root,
            artifact_store=artifact_store,
            manifest=manifest,
            metadata={"cli": True, "target_step": target_step},
        )

        try:
            step_spec = step_spec_class()
            result = step_spec.run(ctx)
            print(f"Completed: {step_name}")
            print(f"   Artifacts: {result.artifacts_written}")
        except Exception as e:
            print(f"Failed: {step_name}")
            print(f"   Error: {e}")
            sys.exit(1)

    print("=" * 70)
    print("Pipeline completed!")
    print()

    print("Artifacts created:")
    manifest_dict = manifest.to_dict()
    steps_dict = manifest_dict.get("steps", {})
    for step_name in execution_order:
        step_info = steps_dict.get(step_name, {})
        if step_info and "artifacts" in step_info:
            for artifact_key in step_info["artifacts"]:
                path = artifact_store.get(ArtifactKey[artifact_key], run_id=run_id)
                if path and path.exists():
                    print(f"   {artifact_key}: {path}")

    print()
    print("Run complete!")
    print()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
