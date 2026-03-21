#!/usr/bin/env python
"""
Custom-op tests for FlexGEMM hashmap operations.

Tests that ``torch.ops.flex_gemm.hashmap_*`` custom-op wrappers:
1. Produce correct results (insert / lookup roundtrip).
2. Are compatible with ``torch.compile`` (no graph breaks).

Ops tested:
    - hashmap_insert_cuda / hashmap_lookup_cuda
    - hashmap_insert_3d_cuda / hashmap_lookup_3d_cuda
    - hashmap_insert_3d_idx_as_val_cuda

Usage::

    srun -G 1 python tests/tensor_ops/test_hashmap.py
"""

from __future__ import annotations

import sys

import torch

# Ensure the custom ops are registered
import flex_gemm  # noqa: F401
from flex_gemm import kernels

from utils import RES, sphere_coords, run_test, print_summary


# ===================================================================
# Correctness tests (eager, no torch.compile)
# ===================================================================


def test_hashmap_insert_lookup_3d_correctness() -> bool:
    """Insert 3D coords, lookup, verify roundtrip values match."""
    print("\n" + "-" * 64)
    print("  hashmap_insert_3d / lookup_3d  correctness")
    print("-" * 64)

    for res in [16, 64, 256]:
        feats, coords, shape = sphere_coords(res, ch=0)
        N = coords.shape[0]
        W, H, D = shape[2], shape[3], shape[4]

        for dtype_key, dtype_val in [
            (torch.uint32, torch.uint32),
            (torch.uint64, torch.uint64),
        ]:
            hashmap_keys = torch.full(
                (2 * N,), torch.iinfo(dtype_key).max,
                dtype=dtype_key, device=coords.device,
            )
            hashmap_values = torch.empty(
                2 * N, dtype=dtype_val, device=coords.device,
            )
            values = torch.randint(
                0, torch.iinfo(dtype_val).max // 2,
                (N,), device=coords.device,
            ).to(dtype_val)

            torch.ops.flex_gemm.hashmap_insert_3d_cuda(
                hashmap_keys, hashmap_values, coords, values, W, H, D,
            )
            result = torch.ops.flex_gemm.hashmap_lookup_3d_cuda(
                hashmap_keys, hashmap_values, coords, W, H, D,
            )

            if not torch.equal(values, result):
                print(f"  res={res} dtype_key={dtype_key} dtype_val={dtype_val}  FAIL")
                return False

    print(f"  [correctness] \033[92mPASS\033[0m")
    return True


def test_hashmap_insert_lookup_correctness() -> bool:
    """Insert flat keys/values, lookup, verify roundtrip."""
    print("\n" + "-" * 64)
    print("  hashmap_insert / lookup  correctness")
    print("-" * 64)

    N = 4096
    for dtype_key, dtype_val in [
        (torch.uint32, torch.uint32),
        (torch.uint64, torch.uint64),
    ]:
        hashmap_keys = torch.full(
            (2 * N,), torch.iinfo(dtype_key).max,
            dtype=dtype_key, device="cuda",
        )
        hashmap_values = torch.empty(
            2 * N, dtype=dtype_val, device="cuda",
        )
        keys = torch.arange(N, device="cuda").to(dtype_key)
        values = torch.randint(
            0, torch.iinfo(dtype_val).max // 2,
            (N,), device="cuda",
        ).to(dtype_val)

        torch.ops.flex_gemm.hashmap_insert_cuda(
            hashmap_keys, hashmap_values, keys, values,
        )
        result = torch.ops.flex_gemm.hashmap_lookup_cuda(
            hashmap_keys, hashmap_values, keys,
        )

        if not torch.equal(values, result):
            print(f"  dtype_key={dtype_key} dtype_val={dtype_val}  FAIL")
            return False

    print(f"  [correctness] \033[92mPASS\033[0m")
    return True


def test_hashmap_insert_3d_idx_as_val_correctness() -> bool:
    """Insert 3D coords using row-index as value, verify."""
    print("\n" + "-" * 64)
    print("  hashmap_insert_3d_idx_as_val  correctness")
    print("-" * 64)

    feats, coords, shape = sphere_coords(RES, ch=0)
    N = coords.shape[0]
    W, H, D = shape[2], shape[3], shape[4]

    hashmap_keys = torch.full(
        (2 * N,), torch.iinfo(torch.uint64).max,
        dtype=torch.uint64, device=coords.device,
    )
    hashmap_values = torch.empty(
        2 * N, dtype=torch.uint32, device=coords.device,
    )

    torch.ops.flex_gemm.hashmap_insert_3d_idx_as_val_cuda(
        hashmap_keys, hashmap_values, coords, W, H, D,
    )
    result = torch.ops.flex_gemm.hashmap_lookup_3d_cuda(
        hashmap_keys, hashmap_values, coords, W, H, D,
    )

    expected = torch.arange(N, device=coords.device, dtype=torch.int64).to(torch.uint32)
    if not torch.equal(result, expected):
        print(f"  FAIL: result != expected")
        return False

    print(f"  [correctness] \033[92mPASS\033[0m")
    return True


# ===================================================================
# torch.compile tests
# ===================================================================


def test_hashmap_lookup_compile() -> bool:
    """Verify hashmap_lookup_3d_cuda works under torch.compile."""
    feats, coords, shape = sphere_coords(RES, ch=0)
    N = coords.shape[0]
    W, H, D = shape[2], shape[3], shape[4]

    # Pre-build hashmap (outside compile boundary)
    hashmap_keys = torch.full(
        (2 * N,), torch.iinfo(torch.uint32).max,
        dtype=torch.uint32, device=coords.device,
    )
    hashmap_values = torch.empty(
        2 * N, dtype=torch.uint32, device=coords.device,
    )
    values = torch.arange(N, device=coords.device, dtype=torch.int64).to(torch.uint32)
    torch.ops.flex_gemm.hashmap_insert_3d_cuda(
        hashmap_keys, hashmap_values, coords, values, W, H, D,
    )

    # The lookup is the compile-able part
    def fn(hk, hv, c):
        return torch.ops.flex_gemm.hashmap_lookup_3d_cuda(hk, hv, c, W, H, D)

    return run_test(
        "hashmap_lookup_3d  compile",
        fn,
        (hashmap_keys, hashmap_values, coords),
    )


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    print("=" * 64)
    print("  FlexGEMM Custom Ops Test: Hashmap")
    print("=" * 64)

    results = {}
    results["insert_lookup_3d"] = test_hashmap_insert_lookup_3d_correctness()
    results["insert_lookup_flat"] = test_hashmap_insert_lookup_correctness()
    results["insert_3d_idx_as_val"] = test_hashmap_insert_3d_idx_as_val_correctness()
    results["lookup_3d_compile"] = test_hashmap_lookup_compile()

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
