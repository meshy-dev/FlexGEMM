#!/usr/bin/env python
"""
CUDA Morton key kernels vs PyTorch reference (meshy_sparse.morton3d_16-compatible).

Usage::

    srun -G 1 python tests/custom_ops/test_morton_keys.py
"""

from __future__ import annotations

import sys

import torch

import flex_gemm  # noqa: F401 — package init / CUDA JIT on first native call
from flex_gemm.ops.spconv.morton_key import (
    morton_keys_3d_16,
    morton_keys_3d_16_reference,
    morton_keys_batched_ncwhd,
    morton_keys_batched_ncwhd_reference,
)


def _pass(name: str) -> None:
    print(f"  {name}  PASS")


def _fail(name: str, msg: str) -> None:
    print(f"  {name}  FAIL  {msg}")


def test_batched_matches_reference() -> bool:
    print("\n" + "-" * 64)
    print("  morton_keys_batched_ncwhd  CUDA vs reference")
    print("-" * 64)
    if not torch.cuda.is_available():
        print("  SKIP (no CUDA)")
        return True

    device = torch.device("cuda")
    torch.manual_seed(0)
    L = 4096
    batch = torch.randint(0, 4, (L,), device=device, dtype=torch.int32)
    w = torch.randint(0, 64, (L,), device=device, dtype=torch.int32)
    h = torch.randint(0, 64, (L,), device=device, dtype=torch.int32)
    d = torch.randint(0, 64, (L,), device=device, dtype=torch.int32)
    coords = torch.stack([batch, w, h, d], dim=1).contiguous()

    for skip_k in (0, 3, 15):
        ref = morton_keys_batched_ncwhd_reference(coords.cpu(), skip_k=skip_k).to(device)
        out = morton_keys_batched_ncwhd(coords, skip_k=skip_k)
        if not torch.equal(out, ref):
            _fail(
                f"batched skip_k={skip_k}",
                f"max_abs={(out - ref).abs().max().item()}",
            )
            return False
    _pass("batched skip_k in {0,3,15}")
    return True


def test_3d_matches_reference_and_batched_batch0() -> bool:
    print("\n" + "-" * 64)
    print("  morton_keys_3d_16  CUDA vs reference / batched b=0")
    print("-" * 64)
    if not torch.cuda.is_available():
        print("  SKIP (no CUDA)")
        return True

    device = torch.device("cuda")
    torch.manual_seed(1)
    L = 2048
    c3 = torch.randint(0, 128, (L, 3), device=device, dtype=torch.int32).contiguous()

    for skip_k in (0, 8):
        ref = morton_keys_3d_16_reference(c3.cpu(), skip_k=skip_k).to(device)
        out = morton_keys_3d_16(c3, skip_k=skip_k)
        if not torch.equal(out, ref):
            _fail(
                f"3d skip_k={skip_k}",
                f"max_abs={(out - ref).abs().max().item()}",
            )
            return False

        zeros = torch.zeros(L, 1, device=device, dtype=torch.int32)
        c4 = torch.cat([zeros, c3], dim=1).contiguous()
        k4 = morton_keys_batched_ncwhd(c4, skip_k=skip_k)
        if not torch.equal(out, k4):
            _fail(
                f"3d vs batched b=0 skip_k={skip_k}",
                "mismatch",
            )
            return False

    _pass("3d + consistency with batched batch=0")
    return True


def main() -> int:
    print("=" * 64)
    print("  FlexGEMM Morton key tests")
    print("=" * 64)
    ok = test_batched_matches_reference() and test_3d_matches_reference_and_batched_batch0()
    print("\n" + "=" * 64)
    print("  OVERALL", "PASS" if ok else "FAIL")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
