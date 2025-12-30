#!/usr/bin/env python3
"""
Check for exact duplicates in NATIX dataset

Usage:
    python3 scripts/check_duplicates.py

Checks:
- Exact duplicates (SHA256 hash)
- Reports which files are duplicates
- Suggests which to keep/remove

2025-12-30: Simple, fast, production-ready
"""
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    print("=" * 70)
    print("NATIX Dataset - Duplicate Checker")
    print("=" * 70)
    
    # Find data directory
    data_root = Path.home() / "data" / "natix_subset"
    
    if not data_root.exists():
        print(f"\n❌ Data not found at: {data_root}")
        print("\nTry these locations:")
        alt_paths = [
            Path.home() / "natix" / "data" / "natix_subset",
            Path.home() / "natix" / "data" / "natix_full",
            Path("/data/natix_subset"),
            Path("/workspace/data/natix_subset"),
        ]
        for p in alt_paths:
            if p.exists():
                print(f"  ✅ Found: {p}")
                data_root = p
                break
            else:
                print(f"  ❌ Not found: {p}")
        
        if not data_root.exists():
            print("\nPlease update data_root in this script or download data first")
            return 1
    
    print(f"\nData directory: {data_root}")
    
    # Find all images
    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_root.rglob(f"*{ext}"))
    
    if not image_files:
        print(f"\n❌ No images found in {data_root}")
        return 1
    
    print(f"Found {len(image_files)} images")
    
    # Hash all files
    print("\nHashing images...")
    hashes = defaultdict(list)
    for path in tqdm(image_files, desc="Progress"):
        try:
            h = hash_file(path)
            hashes[h].append(path)
        except Exception as e:
            print(f"\n⚠️  Error hashing {path}: {e}")
            continue
    
    # Find duplicates
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    
    print("\n" + "=" * 70)
    if duplicates:
        print(f"⚠️  FOUND {len(duplicates)} SETS OF EXACT DUPLICATES")
        print("=" * 70)
        
        total_duplicate_files = sum(len(paths) - 1 for paths in duplicates.values())
        print(f"\nTotal duplicate files: {total_duplicate_files}")
        print(f"Disk space wasted: ~{total_duplicate_files * 0.5:.1f} MB (estimated)")
        
        print("\nDuplicate sets:")
        for i, (h, paths) in enumerate(duplicates.items(), 1):
            print(f"\n{i}. Hash: {h[:16]}... ({len(paths)} copies)")
            for j, p in enumerate(paths):
                marker = "KEEP" if j == 0 else "REMOVE"
                print(f"   [{marker}] {p.relative_to(data_root)}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATION:")
        print("=" * 70)
        print("\n1. Review duplicates above")
        print("2. Remove duplicate files (keep first in each set)")
        print("3. Re-run this script to verify")
        print("4. Regenerate splits.json:")
        print("   python3 scripts/generate_splits.py")
        print("\n⚠️  DO NOT TRAIN WITH DUPLICATES - causes data leakage!")
        
        return 1
    else:
        print("✅ NO EXACT DUPLICATES FOUND")
        print("=" * 70)
        print("\nYour dataset is clean!")
        print("You can proceed with training.")
        return 0


if __name__ == "__main__":
    exit(main())

