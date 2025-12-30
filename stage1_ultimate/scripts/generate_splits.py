#!/usr/bin/env python3
"""
Generate splits.json for NATIX dataset

Creates 4-way stratified split:
- train: 60%
- val_select: 15% (early stopping)
- val_calib: 15% (threshold/calibration)
- val_test: 10% (final eval)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_RATIOS = {
    "train": 0.60,
    "val_select": 0.15,
    "val_calib": 0.15,
    "val_test": 0.10,
}


def _default_data_root() -> Path:
    # Prefer SSH layout
    ssh_root = Path("/workspace/data/natix_subset")
    if ssh_root.exists():
        return ssh_root
    # Local fallback
    return Path.home() / "data" / "natix_subset"


def _load_labels(labels_path: Path) -> list[dict[str, Any]]:
    """
    Load labels.json written by download_full_dataset.py

    Expected format: list of {"filename": "images/xxx.jpg", "label": 0|1}
    """
    with open(labels_path) as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"labels.json invalid or empty: {labels_path}")
    for x in labels:
        if not isinstance(x, dict) or "filename" not in x or "label" not in x:
            raise ValueError(f"labels.json has invalid entry: {x}")
        if int(x["label"]) not in (0, 1):
            raise ValueError(f"labels.json has non-binary label: {x}")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate canonical splits.json for NATIX (binary)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_data_root(),
        help="Data root containing images/ and labels.json",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=None,
        help="Optional explicit path to labels.json (defaults to <data_root>/labels.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "splits.json",
        help="Where to write outputs/splits.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    args = parser.parse_args()

    data_root: Path = args.data_root
    labels_path: Path = args.labels_json or (data_root / "labels.json")
    output_file: Path = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generate NATIX splits (canonical 60/15/15/10)")
    print("=" * 70)
    print(f"Data root:   {data_root}")
    print(f"Labels:      {labels_path}")
    print(f"Output:      {output_file}")
    print("")

    if not data_root.exists():
        print(f"❌ ERROR: data root not found: {data_root}")
        return 1
    if not labels_path.exists():
        print(f"❌ ERROR: labels.json not found: {labels_path}")
        print("Run: python3 scripts/download_full_dataset.py")
        return 1

    labels = _load_labels(labels_path)
    y = np.array([int(x["label"]) for x in labels], dtype=int)

    total = len(labels)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    print(f"Total samples: {total} (class0={n0}, class1={n1})")
    if total == 0:
        print("❌ ERROR: No samples in labels.json")
        return 1

    # First split: train vs rest
    train_ratio = SPLIT_RATIOS["train"]
    rest_ratio = 1.0 - train_ratio
    idx = np.arange(total)
    idx_train, idx_rest = train_test_split(
        idx,
        test_size=rest_ratio,
        random_state=args.seed,
        stratify=y,
    )

    # Split rest into val_select, val_calib, val_test with correct proportions
    # Compute fractions within the rest bucket
    vs = SPLIT_RATIOS["val_select"]
    vc = SPLIT_RATIOS["val_calib"]
    vt = SPLIT_RATIOS["val_test"]
    rest_total = vs + vc + vt
    vs_frac = vs / rest_total
    vc_frac = vc / rest_total
    # vt is remainder

    y_rest = y[idx_rest]
    idx_val_select, idx_tmp = train_test_split(
        idx_rest,
        test_size=(1.0 - vs_frac),
        random_state=args.seed,
        stratify=y_rest,
    )

    y_tmp = y[idx_tmp]
    idx_val_calib, idx_val_test = train_test_split(
        idx_tmp,
        test_size=(1.0 - vc_frac),
        random_state=args.seed,
        stratify=y_tmp,
    )

    def _subset(idxs: np.ndarray) -> list[dict[str, Any]]:
        return [labels[int(i)] for i in idxs]

    splits = {
        "train": _subset(idx_train),
        "val_select": _subset(idx_val_select),
        "val_calib": _subset(idx_val_calib),
        "val_test": _subset(idx_val_test),
    }

    # Verify no overlap + full coverage
    all_ids = (
        set(map(id, splits["train"]))
        | set(map(id, splits["val_select"]))
        | set(map(id, splits["val_calib"]))
        | set(map(id, splits["val_test"]))
    )
    if len(all_ids) != total:
        # This should never happen; kept as a hard fail-fast.
        raise RuntimeError("Split generation produced overlaps or dropped items")

    with open(output_file, "w") as f:
        json.dump(splits, f, indent=2)

    print("")
    print("✅ Splits saved")
    for k in ["train", "val_select", "val_calib", "val_test"]:
        kk = splits[k]
        c0 = sum(1 for x in kk if int(x["label"]) == 0)
        c1 = sum(1 for x in kk if int(x["label"]) == 1)
        print(f"  {k:10s}: {len(kk):6d} (class0={c0}, class1={c1})")

    print("")
    print("Next: run pipeline using this splits file, e.g.")
    print("  python3 scripts/train_cli_v2.py pipeline.phases=[phase1] data.splits_json=outputs/splits.json")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
