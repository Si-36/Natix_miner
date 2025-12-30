#!/usr/bin/env python3
"""
Download NATIX roadwork dataset from HuggingFace to disk (DOWNLOAD-ONLY).

This script intentionally does NOT create splits.
Splits must be generated separately via: `scripts/generate_splits.py`

Why:
- One canonical split policy (e.g. 60/15/15/10) avoids silent inconsistency.
- On SSH GPU boxes, we want a predictable data root: /workspace/data/natix_subset

Output format (binary classification):
<data_root>/
  images/
    <split>_<idx>.jpg
  labels.json

Where labels.json is a list of dicts:
  {"filename": "images/xxx.jpg", "label": 0|1}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME_DEFAULT = "natix-network-org/roadwork"


def _default_output_dir() -> Path:
    # Preferred for rented GPU boxes
    preferred = Path("/workspace/data/natix_subset")
    if preferred.parent.exists():
        return preferred
    # Safe local fallback
    return Path.home() / "data" / "natix_subset"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NATIX roadwork (binary) dataset to disk")
    parser.add_argument("--dataset", default=DATASET_NAME_DEFAULT, help="HuggingFace dataset name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Output directory (data root). Recommended on SSH: /workspace/data/natix_subset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite labels.json even if it exists (images are still written if missing).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"
    labels_path = output_dir / "labels.json"

    print("=" * 70)
    print("NATIX HuggingFace Download (DOWNLOAD-ONLY)")
    print("=" * 70)
    print(f"Dataset:   {args.dataset}")
    print(f"Data root: {output_dir}")
    print("")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if labels_path.exists() and not args.force:
        print(f"✅ labels.json already exists: {labels_path}")
        print("If you want to redownload/rebuild labels, rerun with --force")
        return 0

    print("Downloading from HuggingFace...")
    try:
        dataset = load_dataset(args.dataset)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("")
        print("MANUAL FIX:")
        print(f"1. Go to https://huggingface.co/datasets/{args.dataset}")
        print("2. If private, run: huggingface-cli login")
        return 1

    print(f"Dataset splits: {list(dataset.keys())}")
    print("")

    # Export all splits to a single on-disk pool with labels.json
    labels: list[dict] = []
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"Exporting {split_name}: {len(split_data)} images...")

        for i, example in enumerate(tqdm(split_data, desc=f"  {split_name}")):
            image = example["image"]  # PIL Image
            label = int(example["label"])  # 0 or 1

            image_filename = f"{split_name}_{i:05d}.jpg"
            rel_path = f"images/{image_filename}"
            image_path = images_dir / image_filename

            # Write image to disk
            image.save(image_path)

            labels.append({"filename": rel_path, "label": label})

    # Save labels.json
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    n0 = sum(1 for x in labels if x["label"] == 0)
    n1 = sum(1 for x in labels if x["label"] == 1)

    print("")
    print("✅ Download complete")
    print(f"Images:    {len(labels)} (class0={n0}, class1={n1})")
    print(f"Saved:     {labels_path}")
    print("")
    print("Next step (canonical splits):")
    print(f"  python3 scripts/generate_splits.py --data-root {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
