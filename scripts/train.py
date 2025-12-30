import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    parser = argparse.ArgumentParser(description="Tier 0 Training Pipeline")
    parser.add_argument("--target_step", type=str, default="sweep_thresholds",
        choices=["train_baseline_head", "export_calib_logits", "sweep_thresholds"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--artifact_root", type=str, default="runs")
    return parser.parse_args()

def generate_run_id():
    now = datetime.now()
    return now.strftime("%Y%m%dT%H%M%S")

def run_pipeline(args):
    print("=" * 70)
    print("Tier 0 Training Pipeline")
    print("=" * 70)

    run_id = args.run_id if args.run_id else generate_run_id()
    print(f"Run ID: {run_id}")

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

    print(f"Config: epochs={args.epochs}, synthetic={args.synthetic}")

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(artifact_root)
    print(f"Artifact root: {artifact_root}")

    manifest = RunManifest(run_id=run_id, resolved_config=config)

    target_step = args.target_step
    print(f"Target step: {target_step}")

    try:
        execution_order = resolve_execution_order(target_step, _step_registry)
    except ValueError as e:
        print(f"❌ DAG resolution failed: {e}")
        sys.exit(1)

    print(f"Execution order: {' -> '.join(execution_order)}")

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

            print(f"Completed: {step_name}, artifacts: {result.artifacts_written}")
        except Exception as e:
            print(f"Failed: {step_name}, error: {e}")
            sys.exit(1)

    print("=" * 70)
    print("Pipeline completed!")

    print()
    print("Artifacts:")
    manifest_dict = manifest.to_dict()
    steps_dict = manifest_dict.get("steps", {})
    for step_name in execution_order:
        step_info = steps_dict.get(step_name, {})
        if step_info and "artifacts" in step_info:
            for artifact_key in step_info["artifacts"]:
                path = artifact_store.get(ArtifactKey[artifact_key], run_id=run_id)
                if path and path.exists():
                    print(f"  {artifact_key}: {path}")

    print()
    print("Run complete!")
