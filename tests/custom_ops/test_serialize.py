#!/usr/bin/env python
"""
Custom-op tests for FlexGEMM serialization operations.

Tests that ``torch.ops.flex_gemm.{z_order,hilbert}_{encode,decode}`` and
the high-level ``flex_gemm.ops.serialize.{encode_seq,decode_seq}`` APIs:
1. Produce correct results (encode / decode roundtrip).
2. Are compatible with ``torch.compile`` (no graph breaks).

Usage::

    srun -G 1 python tests/tensor_ops/test_serialize.py
"""

from __future__ import annotations

import sys

import torch
import torch._dynamo as dynamo

import flex_gemm  # noqa: F401 - registers custom ops
from flex_gemm.ops.serialize import encode_seq, decode_seq

from utils import RES, sphere_coords, run_test, print_summary, PASS, FAIL


# ===================================================================
# Correctness tests
# ===================================================================


def test_z_order_roundtrip() -> bool:
    """z_order encode -> decode roundtrip for various resolutions."""
    print("\n" + "-" * 64)
    print("  z_order  encode / decode  correctness")
    print("-" * 64)

    for res in [8, 32, 128, 512]:
        feats, coords, shape = sphere_coords(res, ch=0)
        codes = encode_seq(coords, shape, mode="z_order")
        coords_dec = decode_seq(codes, shape, mode="z_order")

        if not torch.equal(coords, coords_dec):
            print(f"  res={res}  {FAIL}")
            return False

    print(f"  [correctness] {PASS}")
    return True


def test_hilbert_roundtrip() -> bool:
    """hilbert encode -> decode roundtrip for various resolutions."""
    print("\n" + "-" * 64)
    print("  hilbert  encode / decode  correctness")
    print("-" * 64)

    for res in [8, 32, 128, 512]:
        feats, coords, shape = sphere_coords(res, ch=0)
        codes = encode_seq(coords, shape, mode="hilbert")
        coords_dec = decode_seq(codes, shape, mode="hilbert")

        if not torch.equal(coords, coords_dec):
            print(f"  res={res}  {FAIL}")
            return False

    print(f"  [correctness] {PASS}")
    return True


def test_custom_op_z_order_roundtrip() -> bool:
    """Test the low-level custom ops directly (z_order)."""
    print("\n" + "-" * 64)
    print("  torch.ops.flex_gemm.z_order_*  correctness")
    print("-" * 64)

    feats, coords, shape = sphere_coords(RES, ch=0)
    N, C, H, W, D = shape
    bit_length = max(H, W, D).bit_length()

    codes = torch.empty(coords.shape[0], dtype=torch.int64, device=coords.device)
    torch.ops.flex_gemm.z_order_encode(coords, bit_length, codes)
    decoded = torch.ops.flex_gemm.z_order_decode(codes, bit_length)

    if not torch.equal(coords, decoded):
        print(f"  {FAIL}")
        return False

    print(f"  [correctness] {PASS}")
    return True


def test_custom_op_hilbert_roundtrip() -> bool:
    """Test the low-level custom ops directly (hilbert)."""
    print("\n" + "-" * 64)
    print("  torch.ops.flex_gemm.hilbert_*  correctness")
    print("-" * 64)

    feats, coords, shape = sphere_coords(RES, ch=0)
    N, C, H, W, D = shape
    bit_length = max(H, W, D).bit_length()

    codes = torch.empty(coords.shape[0], dtype=torch.int64, device=coords.device)
    torch.ops.flex_gemm.hilbert_encode(coords, bit_length, codes)
    decoded = torch.ops.flex_gemm.hilbert_decode(codes, bit_length)

    if not torch.equal(coords, decoded):
        print(f"  {FAIL}")
        return False

    print(f"  [correctness] {PASS}")
    return True


# ===================================================================
# torch.compile tests
# ===================================================================


def test_z_order_decode_compile() -> bool:
    """Verify z_order_decode works under torch.compile."""
    feats, coords, shape = sphere_coords(RES, ch=0)
    N, C, H, W, D = shape
    bit_length = max(H, W, D).bit_length()

    # Pre-encode
    codes = torch.empty(coords.shape[0], dtype=torch.int64, device=coords.device)
    torch.ops.flex_gemm.z_order_encode(coords, bit_length, codes)

    def fn(codes):
        return torch.ops.flex_gemm.z_order_decode(codes, bit_length)

    return run_test("z_order_decode  compile", fn, (codes,))


def test_hilbert_decode_compile() -> bool:
    """Verify hilbert_decode works under torch.compile."""
    feats, coords, shape = sphere_coords(RES, ch=0)
    N, C, H, W, D = shape
    bit_length = max(H, W, D).bit_length()

    codes = torch.empty(coords.shape[0], dtype=torch.int64, device=coords.device)
    torch.ops.flex_gemm.hilbert_encode(coords, bit_length, codes)

    def fn(codes):
        return torch.ops.flex_gemm.hilbert_decode(codes, bit_length)

    return run_test("hilbert_decode  compile", fn, (codes,))


def test_encode_seq_compile() -> bool:
    """Verify the high-level encode_seq API works under torch.compile.

    Note: encode_seq creates the output tensor and calls mutating ops,
    so we primarily test that the decode path is compilable.
    """
    feats, coords, shape = sphere_coords(RES, ch=0)
    codes = encode_seq(coords, shape, mode="z_order")

    def fn(codes):
        return decode_seq(codes, shape, mode="z_order")

    return run_test("decode_seq(z_order)  compile", fn, (codes,))


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    print("=" * 64)
    print("  FlexGEMM Custom Ops Test: Serialize")
    print("=" * 64)

    results = {}
    results["z_order_roundtrip"] = test_z_order_roundtrip()
    results["hilbert_roundtrip"] = test_hilbert_roundtrip()
    results["z_order_custom_op"] = test_custom_op_z_order_roundtrip()
    results["hilbert_custom_op"] = test_custom_op_hilbert_roundtrip()
    results["z_order_decode_compile"] = test_z_order_decode_compile()
    results["hilbert_decode_compile"] = test_hilbert_decode_compile()
    results["decode_seq_compile"] = test_encode_seq_compile()

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
