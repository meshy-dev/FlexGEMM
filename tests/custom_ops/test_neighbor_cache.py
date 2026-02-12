#!/usr/bin/env python
"""
Custom-op tests for FlexGEMM spconv neighbor-cache operations.

Tests the CUDA custom ops used during neighbor-map construction and
post-processing for masked implicit GEMM:

    - torch.ops.flex_gemm.hashmap_build_submanifold_conv_neighbour_map_cuda
    - torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_1
    - torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_2

Also tests ``SubMConv3dNeighborCache.freeze()`` -> ``SpConvConfig``
conversion, verifying that the frozen config contains valid tensors.

Usage::

    srun -G 1 python tests/tensor_ops/test_neighbor_cache.py
"""

from __future__ import annotations

import sys

import torch

import flex_gemm  # noqa: F401
from flex_gemm.ops.spconv import (
    Algorithm,
    set_algorithm,
    SpConvConfig,
)
from flex_gemm.ops.spconv.submanifold_conv3d import (
    SubMConv3dFunction,
    SubMConv3dNeighborCache,
)

from utils import RES, CH, sphere_coords, run_test, print_summary, PASS, FAIL


# ===================================================================
# Correctness tests
# ===================================================================


def test_neighbor_map_basic() -> bool:
    """Build neighbor map and verify basic structural properties."""
    print("\n" + "-" * 64)
    print("  hashmap_build_submanifold_conv_neighbour_map  correctness")
    print("-" * 64)

    feats, coords, shape = sphere_coords(RES, CH)
    N = coords.shape[0]
    ksize = (3, 3, 3)
    dilation = (1, 1, 1)
    V = ksize[0] * ksize[1] * ksize[2]
    W, H, D = shape[2], shape[3], shape[4]

    from flex_gemm.ops import utils
    from flex_gemm.ops import spconv

    hashmap_keys, hashmap_vals = utils.init_hashmap(
        shape, int(spconv.HASHMAP_RATIO * N), coords.device
    )
    neighbor_map = torch.ops.flex_gemm.hashmap_build_submanifold_conv_neighbour_map_cuda(
        hashmap_keys, hashmap_vals, coords,
        W, H, D,
        ksize[0], ksize[1], ksize[2],
        dilation[0], dilation[1], dilation[2],
    )

    ok = True

    # Shape check
    if neighbor_map.shape != (N, V):
        print(f"  shape: expected ({N}, {V}), got {neighbor_map.shape}  {FAIL}")
        ok = False

    # dtype check
    if neighbor_map.dtype != torch.uint32:
        print(f"  dtype: expected uint32, got {neighbor_map.dtype}  {FAIL}")
        ok = False

    # The center kernel (index V//2) should map each voxel to itself
    INVALID = 0xFFFFFFFF
    center = neighbor_map[:, V // 2]
    expected_self = torch.arange(N, device=coords.device, dtype=torch.int64).to(torch.uint32)
    if not torch.equal(center, expected_self):
        print(f"  center kernel self-map  {FAIL}")
        ok = False

    # Valid entries should be in [0, N)
    neighbor_map_long = neighbor_map.long()
    valid_mask = neighbor_map_long != INVALID
    valid_entries = neighbor_map_long[valid_mask]
    if valid_entries.max() >= N or valid_entries.min() < 0:
        print(f"  valid entries out of range  {FAIL}")
        ok = False

    if ok:
        print(f"  [correctness] {PASS}")
    return ok


def test_post_process_1() -> bool:
    """Test neighbor_map_post_process_for_masked_implicit_gemm_1."""
    print("\n" + "-" * 64)
    print("  neighbor_map_post_process_1  correctness")
    print("-" * 64)

    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    feats, coords, shape = sphere_coords(RES, CH)
    ksize = (3, 3, 3)
    dilation = (1, 1, 1)
    V = ksize[0] * ksize[1] * ksize[2]
    N = coords.shape[0]

    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)

    ok = True

    # Check that the expected fields exist
    if not hasattr(cache, "gray_code"):
        print(f"  missing gray_code  {FAIL}")
        ok = False
    if not hasattr(cache, "sorted_idx"):
        print(f"  missing sorted_idx  {FAIL}")
        ok = False
    if not hasattr(cache, "valid_signal_i"):
        print(f"  missing valid_signal_i  {FAIL}")
        ok = False
    if not hasattr(cache, "valid_signal_seg"):
        print(f"  missing valid_signal_seg  {FAIL}")
        ok = False

    if not ok:
        return False

    # Shape checks
    if cache["gray_code"].shape != (N,):
        print(f"  gray_code shape: {cache['gray_code'].shape}  {FAIL}")
        ok = False
    if cache["sorted_idx"].shape != (N,):
        print(f"  sorted_idx shape: {cache['sorted_idx'].shape}  {FAIL}")
        ok = False
    if cache["valid_signal_seg"].shape != (V + 1,):
        print(f"  valid_signal_seg shape: {cache['valid_signal_seg'].shape}  {FAIL}")
        ok = False

    # sorted_idx should be a permutation of [0, N)
    sorted_vals = cache["sorted_idx"].sort().values
    expected = torch.arange(N, device=coords.device, dtype=cache["sorted_idx"].dtype)
    if not torch.equal(sorted_vals, expected):
        print(f"  sorted_idx not a valid permutation  {FAIL}")
        ok = False

    if ok:
        print(f"  [correctness] {PASS}")
    return ok


def test_post_process_2() -> bool:
    """Test neighbor_map_post_process_for_masked_implicit_gemm_2."""
    print("\n" + "-" * 64)
    print("  neighbor_map_post_process_2  correctness")
    print("-" * 64)

    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    feats, coords, shape = sphere_coords(RES, CH)
    ksize = (3, 3, 3)
    dilation = (1, 1, 1)

    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)

    ok = True
    for block_size in [32, 64, 128, 256]:
        valid_kernel, valid_kernel_seg = (
            torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_2(
                cache["gray_code"], cache["sorted_idx"], block_size,
            )
        )

        N = coords.shape[0]
        num_blocks = (N + block_size - 1) // block_size

        if valid_kernel_seg.shape != (num_blocks + 1,):
            print(f"  block_size={block_size} seg shape: {valid_kernel_seg.shape}  {FAIL}")
            ok = False

        # valid_kernel entries should be non-negative
        if valid_kernel.numel() > 0 and valid_kernel.min() < 0:
            print(f"  block_size={block_size} negative kernel idx  {FAIL}")
            ok = False

    if ok:
        print(f"  [correctness] {PASS}")
    return ok


def test_freeze_spconv_config() -> bool:
    """Test SubMConv3dNeighborCache.freeze() -> SpConvConfig."""
    print("\n" + "-" * 64)
    print("  SubMConv3dNeighborCache.freeze()  correctness")
    print("-" * 64)

    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    feats, coords, shape = sphere_coords(RES, CH)
    ksize = (3, 3, 3)
    dilation = (1, 1, 1)

    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, ksize, dilation)
    config = cache.freeze(splitk=1, algorithm="masked_implicit_gemm_splitk")

    ok = True

    if not isinstance(config, SpConvConfig):
        print(f"  not a SpConvConfig  {FAIL}")
        return False

    # Check all fields
    if config.neighbor_map.shape[0] != coords.shape[0]:
        print(f"  neighbor_map N mismatch  {FAIL}")
        ok = False

    if len(config.valid_kernels) != len(config.block_sizes):
        print(f"  valid_kernels length mismatch  {FAIL}")
        ok = False

    if len(config.valid_kernel_segs) != len(config.block_sizes):
        print(f"  valid_kernel_segs length mismatch  {FAIL}")
        ok = False

    if config.block_sizes != [32, 64, 128, 256]:
        print(f"  unexpected block_sizes: {config.block_sizes}  {FAIL}")
        ok = False

    if config.splitk != 1:
        print(f"  unexpected splitk: {config.splitk}  {FAIL}")
        ok = False

    if config.algorithm != "masked_implicit_gemm_splitk":
        print(f"  unexpected algorithm: {config.algorithm}  {FAIL}")
        ok = False

    if ok:
        print(f"  [correctness] {PASS}")
    return ok


# ===================================================================
# torch.compile test for neighbor_map_post_process ops
# ===================================================================


def test_post_process_compile() -> bool:
    """Verify post-process ops work under torch.compile.

    These ops have data-dependent output sizes (dynamic SymInt),
    testing that register_fake with new_dynamic_size() works.
    """
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    feats, coords, shape = sphere_coords(RES, CH)
    ksize = (3, 3, 3)
    dilation = (1, 1, 1)
    N = coords.shape[0]
    W, H, D = shape[2], shape[3], shape[4]

    from flex_gemm.ops import utils, spconv

    hashmap_keys, hashmap_vals = utils.init_hashmap(
        shape, int(spconv.HASHMAP_RATIO * N), coords.device,
    )
    neighbor_map = torch.ops.flex_gemm.hashmap_build_submanifold_conv_neighbour_map_cuda(
        hashmap_keys, hashmap_vals, coords,
        W, H, D,
        ksize[0], ksize[1], ksize[2],
        dilation[0], dilation[1], dilation[2],
    )

    def fn(neighbor_map):
        gray_code, sorted_idx, vsi, vso, vss = (
            torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_1(
                neighbor_map
            )
        )
        vk, vks = (
            torch.ops.flex_gemm.neighbor_map_post_process_for_masked_implicit_gemm_2(
                gray_code, sorted_idx, 64,
            )
        )
        return gray_code, sorted_idx, vk

    return run_test(
        "post_process_1 + post_process_2  compile",
        fn,
        (neighbor_map,),
    )


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    print("=" * 64)
    print("  FlexGEMM Custom Ops Test: Neighbor Cache")
    print("=" * 64)

    results = {}
    results["neighbor_map_basic"] = test_neighbor_map_basic()
    results["post_process_1"] = test_post_process_1()
    results["post_process_2"] = test_post_process_2()
    results["freeze_config"] = test_freeze_spconv_config()
    results["post_process_compile"] = test_post_process_compile()

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
