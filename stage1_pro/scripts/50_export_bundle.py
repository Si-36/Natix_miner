#!/usr/bin/env python3
"""
Phase 1: Export deployment bundle with active_exit_policy pointer.

Creates bundle.json with:
- active_exit_policy: "softmax" (Phase 1)
- policy_file: path to thresholds.json
- model_file: path to model_best.pth
- splits_file: path to splits.json
- config_file: path to config.json
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess


def create_bundle(
    output_dir: str,
    model_path: str = "model_best.pth",
    thresholds_path: str = "thresholds.json",
    splits_path: str = "splits.json",
    config_path: str = "config.json",
    metrics_path: str = "metrics.csv",
    description: str = "Stage-1 Pro Phase 1 Baseline Training",
):
    """
    Create bundle.json manifest for deployment.

    Phase 1: active_exit_policy = "softmax"
    """
    output_path = Path(output_dir)

    # Get git commit if available
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        git_commit = result.stdout.strip()
    except:
        pass

    bundle = {
        "active_exit_policy": "softmax",  # Phase 1 only
        "policy_file": thresholds_path,
        "model_file": model_path,
        "splits_file": splits_path,
        "config_file": config_path,
        "metrics_file": metrics_path,
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit,
        "description": description,
    }

    # Validate against schema
    schema_path = Path(__file__).parent.parent / "schemas" / "bundle.schema.json"
    if schema_path.exists():
        import jsonschema

        with open(schema_path) as f:
            schema = json.load(f)

        jsonschema.validate(bundle, schema)
        print(f"✅ Bundle validated against {schema_path}")

    # Save bundle
    bundle_file = output_path / "bundle.json"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(bundle_file, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"\n✅ Bundle manifest created: {bundle_file}")
    print(f"   Active exit policy: {bundle['active_exit_policy']}")
    print(f"   Policy file: {bundle['policy_file']}")
    print(f"   Model file: {bundle['model_file']}")

    return bundle_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="model_best.pth")
    parser.add_argument("--thresholds_path", type=str, default="thresholds.json")
    parser.add_argument("--splits_path", type=str, default="splits.json")
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--metrics_path", type=str, default="metrics.csv")
    parser.add_argument(
        "--description", type=str, default="Stage-1 Pro Phase 1 Baseline Training"
    )

    args = parser.parse_args()

    create_bundle(
        output_dir=args.output_dir,
        model_path=args.model_path,
        thresholds_path=args.thresholds_path,
        splits_path=args.splits_path,
        config_path=args.config_path,
        metrics_path=args.metrics_path,
        description=args.description,
    )


if __name__ == "__main__":
    main()
