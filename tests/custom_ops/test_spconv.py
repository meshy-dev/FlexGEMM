#!/usr/bin/env python
"""
Custom-op tests for FlexGEMM sparse convolution compiled path.

Tests the ``torch.ops.flex_gemm.sparse_conv_masked_*`` custom ops and
the ``SubMConv3dCompiledFunction`` autograd function that uses them:

    - sparse_conv_masked_fwd / sparse_conv_masked_bwd
    - sparse_conv_masked_splitk_fwd / sparse_conv_masked_splitk_bwd
    - sparse_submanifold_conv3d(config=SpConvConfig) public API

Checks:
    1. Compiled path output matches legacy (SubMConv3dFunction) output.
    2. No graph breaks under ``torch.compile``.
    3. ``fullgraph=True`` compiles successfully.
    4. Backward gradient correctness (compiled vs eager).

Usage::

    srun -G 1 python tests/tensor_ops/test_spconv.py
"""

from __future__ import annotations

import sys

import torch
import torch._dynamo as dynamo

import flex_gemm  # noqa: F401
from flex_gemm.ops.spconv import (
    Algorithm,
    set_algorithm,
    SpConvConfig,
    sparse_submanifold_conv3d,
)
from flex_gemm.ops.spconv.submanifold_conv3d import (
    SubMConv3dFunction,
)

from utils import RES, CH, sphere_coords, run_test, print_summary, PASS, FAIL


# ===================================================================
# Helpers
# ===================================================================


def _make_spconv_data(res=RES, ch=CH):
    """Build test data and a frozen SpConvConfig."""
    feats, coords, shape = sphere_coords(res, ch, dtype=torch.float32)
    weight = torch.randn(ch, 3, 3, 3, ch, device="cuda")
    bias = torch.randn(ch, device="cuda")

    # Build neighbor cache via legacy path
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    with torch.no_grad():
        out_legacy, neighbor_cache = sparse_submanifold_conv3d(
            feats, coords, shape, weight, bias,
        )

    # Freeze into compile-friendly config
    config = neighbor_cache.freeze(
        splitk=1, algorithm="masked_implicit_gemm_splitk",
    )

    return feats, coords, shape, weight, bias, config, out_legacy, neighbor_cache


# ===================================================================
# Correctness tests
# ===================================================================


def test_compiled_fwd_correctness() -> bool:
    """Compiled forward output matches legacy forward output."""
    print("\n" + "-" * 64)
    print("  spconv compiled fwd  correctness (vs legacy)")
    print("-" * 64)

    feats, coords, shape, weight, bias, config, out_legacy, _ = _make_spconv_data()

    with torch.no_grad():
        out_compiled = sparse_submanifold_conv3d(
            feats, weight=weight, bias=bias, config=config,
        )

    diff = (out_legacy.float() - out_compiled.float()).abs().max().item()
    # Use relaxed tolerance: split-K accumulation order can differ slightly
    if torch.allclose(out_legacy, out_compiled, rtol=1e-3, atol=1e-3):
        print(f"  max diff = {diff:.6e}  {PASS}")
        return True
    else:
        print(f"  max diff = {diff:.6e}  {FAIL}")
        return False


def test_compiled_bwd_correctness() -> bool:
    """Compiled backward gradients match legacy backward gradients."""
    print("\n" + "-" * 64)
    print("  spconv compiled bwd  correctness (vs legacy)")
    print("-" * 64)

    feats, coords, shape, weight, bias, config, _, neighbor_cache = _make_spconv_data()

    # Legacy backward (use SubMConv3dFunction.apply which sets up autograd)
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    feats_e = feats.detach().clone().requires_grad_(True)
    weight_e = weight.detach().clone().requires_grad_(True)
    bias_e = bias.detach().clone().requires_grad_(True)
    out_e, _ = SubMConv3dFunction.apply(
        feats_e, coords, shape, neighbor_cache, weight_e, bias_e, (1, 1, 1),
    )
    out_e.sum().backward()

    # Compiled path backward
    feats_c = feats.detach().clone().requires_grad_(True)
    weight_c = weight.detach().clone().requires_grad_(True)
    bias_c = bias.detach().clone().requires_grad_(True)
    out_c = sparse_submanifold_conv3d(
        feats_c, weight=weight_c, bias=bias_c, config=config,
    )
    out_c.sum().backward()

    ok = True
    for name, g_e, g_c in [
        ("grad_feats", feats_e.grad, feats_c.grad),
        ("grad_weight", weight_e.grad, weight_c.grad),
        ("grad_bias", bias_e.grad, bias_c.grad),
    ]:
        if g_e is None and g_c is None:
            continue
        if g_e is None or g_c is None:
            print(f"  {name}: one is None  {FAIL}")
            ok = False
            continue
        diff = (g_e.float() - g_c.float()).abs().max().item()
        tag = PASS if diff < 1e-2 else FAIL
        print(f"  {name} max diff = {diff:.6e}  {tag}")
        if diff >= 1e-2:
            ok = False

    return ok


def test_masked_fwd_correctness() -> bool:
    """Test non-split-K masked forward (sparse_conv_masked_fwd)."""
    print("\n" + "-" * 64)
    print("  sparse_conv_masked_fwd  correctness")
    print("-" * 64)

    feats, coords, shape = sphere_coords(RES, CH, dtype=torch.float32)
    weight = torch.randn(CH, 3, 3, 3, CH, device="cuda")
    bias = torch.randn(CH, device="cuda")

    # Build neighbor cache
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
        coords, shape, (3, 3, 3), (1, 1, 1),
    )
    out_legacy = SubMConv3dFunction._sparse_submanifold_conv_forward(
        feats, neighbor_cache, weight, bias,
    )

    # Compiled path (non-split-K)
    config = neighbor_cache.freeze(splitk=1, algorithm="masked_implicit_gemm")
    out_compiled = sparse_submanifold_conv3d(
        feats, weight=weight, bias=bias, config=config,
    )

    if torch.allclose(out_legacy, out_compiled, rtol=1e-4, atol=1e-4):
        diff = (out_legacy.float() - out_compiled.float()).abs().max().item()
        print(f"  max diff = {diff:.6e}  {PASS}")
        return True
    else:
        diff = (out_legacy.float() - out_compiled.float()).abs().max().item()
        print(f"  max diff = {diff:.6e}  {FAIL}")
        return False


# ===================================================================
# torch.compile tests
# ===================================================================


def test_spconv_compiled_fwd_compile() -> bool:
    """sparse_submanifold_conv3d compiled path under torch.compile."""
    feats, _, _, weight, bias, config, _, _ = _make_spconv_data()

    def fn(feats, weight, bias):
        return sparse_submanifold_conv3d(
            feats, weight=weight, bias=bias, config=config,
        )

    return run_test(
        "spconv compiled fwd  compile",
        fn,
        (feats, weight, bias),
    )


def test_spconv_compiled_bwd_compile() -> bool:
    """sparse_submanifold_conv3d compiled fwd+bwd under torch.compile."""
    feats, _, _, weight, bias, config, _, _ = _make_spconv_data()
    feats = feats.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    bias = bias.detach().requires_grad_(True)

    def fn(feats, weight, bias):
        return sparse_submanifold_conv3d(
            feats, weight=weight, bias=bias, config=config,
        )

    return run_test(
        "spconv compiled fwd+bwd  compile",
        fn,
        (feats, weight, bias),
        test_bwd=True,
        bwd_input_idx=0,
    )


def test_spconv_masked_fwd_compile() -> bool:
    """Non-split-K masked forward under torch.compile."""
    feats, coords, shape = sphere_coords(RES, CH, dtype=torch.float32)
    weight = torch.randn(CH, 3, 3, 3, CH, device="cuda")
    bias = torch.randn(CH, device="cuda")

    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
    neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
        coords, shape, (3, 3, 3), (1, 1, 1),
    )
    config = neighbor_cache.freeze(splitk=1, algorithm="masked_implicit_gemm")

    def fn(feats, weight, bias):
        return sparse_submanifold_conv3d(
            feats, weight=weight, bias=bias, config=config,
        )

    return run_test(
        "sparse_conv_masked fwd  compile",
        fn,
        (feats, weight, bias),
    )


# ===================================================================
# Direct custom op tests
# ===================================================================


def test_sparse_conv_masked_splitk_fwd_op() -> bool:
    """Test torch.ops.flex_gemm.sparse_conv_masked_splitk_fwd directly."""
    print("\n" + "-" * 64)
    print("  torch.ops.flex_gemm.sparse_conv_masked_splitk_fwd  correctness")
    print("-" * 64)

    feats, _, _, weight, bias, config, out_legacy, _ = _make_spconv_data()
    Co, Kw, Kh, Kd, Ci = weight.shape
    V = Kw * Kh * Kd

    out_op = torch.ops.flex_gemm.sparse_conv_masked_splitk_fwd(
        feats,
        weight.reshape(Co, V, Ci),
        bias,
        config.neighbor_map,
        config.sorted_idx,
        config.valid_kernels,
        config.valid_kernel_segs,
        config.block_sizes,
        config.splitk,
    )

    diff = (out_legacy.float() - out_op.float()).abs().max().item()
    # Relaxed tolerance: split-K accumulation order can differ slightly
    if torch.allclose(out_legacy, out_op, rtol=1e-3, atol=1e-3):
        print(f"  max diff = {diff:.6e}  {PASS}")
        return True
    else:
        print(f"  max diff = {diff:.6e}  {FAIL}")
        return False


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    print("=" * 64)
    print("  FlexGEMM Custom Ops Test: Sparse Convolution")
    print("=" * 64)

    results = {}

    # Correctness tests
    results["compiled_fwd_correctness"] = test_compiled_fwd_correctness()
    results["compiled_bwd_correctness"] = test_compiled_bwd_correctness()
    results["masked_fwd_correctness"] = test_masked_fwd_correctness()
    results["splitk_fwd_op"] = test_sparse_conv_masked_splitk_fwd_op()

    # torch.compile tests
    results["compiled_fwd_compile"] = test_spconv_compiled_fwd_compile()
    results["compiled_bwd_compile"] = test_spconv_compiled_bwd_compile()
    results["masked_fwd_compile"] = test_spconv_masked_fwd_compile()

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
