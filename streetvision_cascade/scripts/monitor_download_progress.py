#!/usr/bin/env python3
"""
Monitor Hugging Face snapshot_download progress for large models.

Why: Hugging Face creates sparse `.incomplete` files (apparent size is huge),
so `ls -lh` can look "stuck". This script reports REAL downloaded bytes
using filesystem allocated blocks (st_blocks * 512).

Works without any external dependencies.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_GLM_DOWNLOAD_DIR = (
    "/home/sina/projects/miner_b/streetvision_cascade/"
    "models/stage3_glm/GLM-4.6V-Flash/.cache/huggingface/download"
)
DEFAULT_MOLMO_DOWNLOAD_DIR = (
    "/home/sina/projects/miner_b/streetvision_cascade/"
    "models/stage3_molmo/Molmo2-8B/.cache/huggingface/download"
)


@dataclass(frozen=True)
class FileSnap:
    path: str
    physical_bytes: int
    apparent_bytes: int


def _fmt_bytes(n: int) -> str:
    # Human readable (base2-ish but simple)
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f}{u}"
        v /= 1024.0
    return f"{v:.2f}TB"


def _collect_incomplete(download_dir: str) -> List[FileSnap]:
    pattern = os.path.join(download_dir, "*.incomplete")
    out: List[FileSnap] = []
    for p in sorted(glob.glob(pattern)):
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        physical = st.st_blocks * 512  # real bytes allocated
        out.append(FileSnap(path=p, physical_bytes=physical, apparent_bytes=st.st_size))
    return out


def _summarize(snaps: List[FileSnap]) -> Tuple[int, int]:
    phys = sum(s.physical_bytes for s in snaps)
    app = sum(s.apparent_bytes for s in snaps)
    return phys, app


def _basename_short(p: str, max_len: int = 40) -> str:
    b = os.path.basename(p)
    if len(b) <= max_len:
        return b
    return b[: (max_len - 3)] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor HF .incomplete download progress")
    parser.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        default=[],
        help="Download directory containing *.incomplete files (repeatable)",
    )
    parser.add_argument("--glm", action="store_true", help="Monitor GLM stage3 directory")
    parser.add_argument("--molmo", action="store_true", help="Monitor Molmo stage3 directory")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between updates")
    parser.add_argument("--top", type=int, default=8, help="Show top N largest incomplete files")
    args = parser.parse_args()

    dirs: List[str] = []
    if args.glm:
        dirs.append(DEFAULT_GLM_DOWNLOAD_DIR)
    if args.molmo:
        dirs.append(DEFAULT_MOLMO_DOWNLOAD_DIR)
    dirs.extend(args.dirs)

    if not dirs:
        # Default: try GLM + Molmo
        dirs = [DEFAULT_GLM_DOWNLOAD_DIR, DEFAULT_MOLMO_DOWNLOAD_DIR]

    # Normalize and keep only existing dirs
    resolved: List[str] = []
    for d in dirs:
        p = Path(d).expanduser()
        if p.is_dir():
            resolved.append(str(p))
    if not resolved:
        print("No valid download directories found.")
        print("Pass --dir <path> or use --glm / --molmo.")
        return 2

    last: Dict[str, FileSnap] = {}
    last_time = time.time()

    while True:
        now = time.time()
        dt = max(now - last_time, 1e-6)
        last_time = now

        print("\n" + "=" * 90)
        print(time.strftime("%Y-%m-%d %H:%M:%S"))

        for d in resolved:
            snaps = _collect_incomplete(d)
            phys_sum, app_sum = _summarize(snaps)
            print(f"\nDIR: {d}")
            print(f"  Incomplete files: {len(snaps)}")
            print(f"  Downloaded (real): {_fmt_bytes(phys_sum)}")
            print(f"  Total remaining (apparent): {_fmt_bytes(app_sum)}")

            # Show top N by physical bytes
            snaps_sorted = sorted(snaps, key=lambda s: s.physical_bytes, reverse=True)
            shown = snaps_sorted[: max(args.top, 0)]
            if shown:
                print("  Top files (real bytes, with delta and speed):")
                for s in shown:
                    prev = last.get(s.path)
                    delta = s.physical_bytes - (prev.physical_bytes if prev else 0)
                    speed = delta / dt
                    name = _basename_short(s.path)
                    print(
                        f"   - {name:40}  {_fmt_bytes(s.physical_bytes):>10}  "
                        f"Δ{_fmt_bytes(delta):>10}  ~{_fmt_bytes(int(speed))}/s"
                    )

            # Update last snapshots for this dir
            for s in snaps:
                last[s.path] = s

        time.sleep(max(args.interval, 0.2))


if __name__ == "__main__":
    raise SystemExit(main())


