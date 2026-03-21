#!/usr/bin/env python
"""
Run all FlexGEMM custom-op tensor tests.

Usage::

    srun -G 1 python tests/tensor_ops/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch

_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"

TEST_DIR = Path(__file__).resolve().parent
TEST_FILES = [
    "test_hashmap.py",
    "test_serialize.py",
    "test_grid_sample.py",
    "test_neighbor_cache.py",
    "test_spconv.py",
]


def main() -> int:
    device_name = (
        torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
    )
    print("=" * 72)
    print("  FlexGEMM Custom Ops -- Full Test Suite")
    print("=" * 72)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.version.cuda}")
    print(f"  Device   : {device_name}")
    print("=" * 72)

    results: dict[str, tuple[int, float]] = {}

    for fname in TEST_FILES:
        fpath = TEST_DIR / fname
        if not fpath.exists():
            print(f"\n  WARNING: {fname} not found, skipping.")
            results[fname] = (1, 0.0)
            continue

        print(f"\n{'#' * 72}")
        print(f"# Running {fname}")
        print(f"{'#' * 72}\n")

        t0 = time.time()
        ret = subprocess.run(
            [sys.executable, str(fpath)],
            cwd=str(TEST_DIR),
        )
        elapsed = time.time() - t0
        results[fname] = (ret.returncode, elapsed)

    # Final summary
    print()
    print("=" * 72)
    print("  OVERALL SUMMARY")
    print("=" * 72)

    n_pass = 0
    for fname, (rc, elapsed) in results.items():
        if rc == 0:
            tag = f"{_GREEN}PASS{_RESET}"
            n_pass += 1
        else:
            tag = f"{_RED}FAIL{_RESET}"
        print(f"  {fname:<40} {tag}  ({elapsed:.1f}s)")

    print("-" * 72)
    print(f"  {n_pass}/{len(results)} test files passed")
    print("=" * 72)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
