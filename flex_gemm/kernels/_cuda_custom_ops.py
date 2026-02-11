"""
Custom op wrappers for FlexGEMM CUDA extension functions.

Registers all pybind11 CUDA extension ops as ``torch.library`` custom ops
with FakeTensor implementations so that ``torch.compile`` can trace through
them without graph breaks.

This module is imported at package init time
(``flex_gemm.kernels.__init__``) to ensure the ops are registered.  They
can then be called via::

    torch.ops.flex_gemm.<op_name>(...)
"""

from typing import Tuple

import torch
from torch import Tensor

from . import cuda as _C


# ============================================================
# Hash ops
# ============================================================


@torch.library.custom_op(
    "flex_gemm::hashmap_insert_cuda",
    mutates_args=("hashmap_keys", "hashmap_values"),
)
def hashmap_insert_cuda(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    keys: Tensor,
    values: Tensor,
) -> None:
    """Insert *keys* / *values* into the hashmap."""
    _C.hashmap_insert_cuda(hashmap_keys, hashmap_values, keys, values)


@hashmap_insert_cuda.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    keys: Tensor,
    values: Tensor,
) -> None:
    pass


# ------------------------------------------------------------------


@torch.library.custom_op("flex_gemm::hashmap_lookup_cuda", mutates_args=())
def hashmap_lookup_cuda(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    keys: Tensor,
) -> Tensor:
    """Lookup *keys* in the hashmap, returning corresponding values."""
    return _C.hashmap_lookup_cuda(hashmap_keys, hashmap_values, keys)


@hashmap_lookup_cuda.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    keys: Tensor,
) -> Tensor:
    return hashmap_values.new_empty(keys.shape[0])


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::hashmap_insert_3d_cuda",
    mutates_args=("hashmap_keys", "hashmap_values"),
)
def hashmap_insert_3d_cuda(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    coords: Tensor,
    values: Tensor,
    W: int,
    H: int,
    D: int,
) -> None:
    """Insert 3-D coordinates into the hashmap."""
    _C.hashmap_insert_3d_cuda(
        hashmap_keys, hashmap_values, coords, values, W, H, D
    )


@hashmap_insert_3d_cuda.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    coords: Tensor,
    values: Tensor,
    W: int,
    H: int,
    D: int,
) -> None:
    pass


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::hashmap_lookup_3d_cuda", mutates_args=()
)
def hashmap_lookup_3d_cuda(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    coords: Tensor,
    W: int,
    H: int,
    D: int,
) -> Tensor:
    """Lookup 3-D coordinates in the hashmap."""
    return _C.hashmap_lookup_3d_cuda(
        hashmap_keys, hashmap_values, coords, W, H, D
    )


@hashmap_lookup_3d_cuda.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    coords: Tensor,
    W: int,
    H: int,
    D: int,
) -> Tensor:
    return hashmap_values.new_empty(coords.shape[0])


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::hashmap_insert_3d_idx_as_val_cuda",
    mutates_args=("hashmap_keys", "hashmap_values"),
)
def hashmap_insert_3d_idx_as_val_cuda(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    coords: Tensor,
    W: int,
    H: int,
    D: int,
) -> None:
    """Insert 3-D coordinates into the hashmap using the row index as value."""
    _C.hashmap_insert_3d_idx_as_val_cuda(
        hashmap_keys, hashmap_values, coords, W, H, D
    )


@hashmap_insert_3d_idx_as_val_cuda.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_values: Tensor,
    coords: Tensor,
    W: int,
    H: int,
    D: int,
) -> None:
    pass


# ============================================================
# Serialization ops
# ============================================================


@torch.library.custom_op(
    "flex_gemm::z_order_encode", mutates_args=("codes",)
)
def z_order_encode(
    coords: Tensor,
    bit_length: int,
    codes: Tensor,
) -> None:
    """Z-order encode 3-D coordinates (writes into *codes* in-place)."""
    _C.z_order_encode(coords, bit_length, codes)


@z_order_encode.register_fake
def _(coords: Tensor, bit_length: int, codes: Tensor) -> None:
    pass


# ------------------------------------------------------------------


@torch.library.custom_op("flex_gemm::z_order_decode", mutates_args=())
def z_order_decode(
    codes: Tensor,
    bit_length: int,
) -> Tensor:
    """Z-order decode to ``[N, 4]`` (b, x, y, z) coordinates."""
    return _C.z_order_decode(codes, bit_length)


@z_order_decode.register_fake
def _(codes: Tensor, bit_length: int) -> Tensor:
    return torch.empty(codes.shape[0], 4, dtype=torch.int32, device=codes.device)


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::hilbert_encode", mutates_args=("codes",)
)
def hilbert_encode(
    coords: Tensor,
    bit_length: int,
    codes: Tensor,
) -> None:
    """Hilbert encode 3-D coordinates (writes into *codes* in-place)."""
    _C.hilbert_encode(coords, bit_length, codes)


@hilbert_encode.register_fake
def _(coords: Tensor, bit_length: int, codes: Tensor) -> None:
    pass


# ------------------------------------------------------------------


@torch.library.custom_op("flex_gemm::hilbert_decode", mutates_args=())
def hilbert_decode(
    codes: Tensor,
    bit_length: int,
) -> Tensor:
    """Hilbert decode to ``[N, 4]`` (b, x, y, z) coordinates."""
    return _C.hilbert_decode(codes, bit_length)


@hilbert_decode.register_fake
def _(codes: Tensor, bit_length: int) -> Tensor:
    return torch.empty(codes.shape[0], 4, dtype=torch.int32, device=codes.device)


# ============================================================
# Grid-sample ops
# ============================================================


@torch.library.custom_op(
    "flex_gemm::hashmap_build_grid_sample_3d_nearest_neighbor_map",
    mutates_args=("hashmap_keys", "hashmap_vals"),
)
def hashmap_build_grid_sample_3d_nearest_neighbor_map(
    hashmap_keys: Tensor,
    hashmap_vals: Tensor,
    coords: Tensor,
    grid: Tensor,
    W: int,
    H: int,
    D: int,
) -> Tensor:
    """Build ``[B, L]`` uint32 nearest-neighbor index map."""
    return _C.hashmap_build_grid_sample_3d_nearest_neighbor_map(
        hashmap_keys, hashmap_vals, coords, grid, W, H, D
    )


@hashmap_build_grid_sample_3d_nearest_neighbor_map.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_vals: Tensor,
    coords: Tensor,
    grid: Tensor,
    W: int,
    H: int,
    D: int,
) -> Tensor:
    B, L = grid.shape[:2]
    return torch.empty(B, L, dtype=torch.uint32, device=grid.device)


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight",
    mutates_args=("hashmap_keys", "hashmap_vals"),
)
def hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
    hashmap_keys: Tensor,
    hashmap_vals: Tensor,
    coords: Tensor,
    grid: Tensor,
    W: int,
    H: int,
    D: int,
) -> Tuple[Tensor, Tensor]:
    """Build ``[B, L, 8]`` trilinear neighbor indices and weights."""
    return _C.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
        hashmap_keys, hashmap_vals, coords, grid, W, H, D
    )


@hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_vals: Tensor,
    coords: Tensor,
    grid: Tensor,
    W: int,
    H: int,
    D: int,
) -> Tuple[Tensor, Tensor]:
    B, L = grid.shape[:2]
    return (
        torch.empty(B, L, 8, dtype=torch.uint32, device=grid.device),
        grid.new_empty(B, L, 8),
    )


# ============================================================
# Sparse-convolution ops
# ============================================================


@torch.library.custom_op(
    "flex_gemm::hashmap_build_submanifold_conv_neighbour_map_cuda",
    mutates_args=("hashmap_keys", "hashmap_vals"),
)
def hashmap_build_submanifold_conv_neighbour_map_cuda(
    hashmap_keys: Tensor,
    hashmap_vals: Tensor,
    coords: Tensor,
    W: int,
    H: int,
    D: int,
    Kw: int,
    Kh: int,
    Kd: int,
    Dw: int,
    Dh: int,
    Dd: int,
) -> Tensor:
    """Build ``[M, Kw*Kh*Kd]`` uint32 submanifold convolution neighbor map."""
    return _C.hashmap_build_submanifold_conv_neighbour_map_cuda(
        hashmap_keys, hashmap_vals, coords, W, H, D, Kw, Kh, Kd, Dw, Dh, Dd
    )


@hashmap_build_submanifold_conv_neighbour_map_cuda.register_fake
def _(
    hashmap_keys: Tensor,
    hashmap_vals: Tensor,
    coords: Tensor,
    W: int,
    H: int,
    D: int,
    Kw: int,
    Kh: int,
    Kd: int,
    Dw: int,
    Dh: int,
    Dd: int,
) -> Tensor:
    M = coords.shape[0]
    V = Kw * Kh * Kd
    return torch.empty(M, V, dtype=torch.uint32, device=coords.device)


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::neighbor_map_post_process_for_masked_implicit_gemm_1",
    mutates_args=(),
)
def neighbor_map_post_process_for_masked_implicit_gemm_1(
    neighbor_map: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Post-process neighbor map into gray code, sorted indices, and valid signals.

    Returns:
        gray_code ``[N]`` int32,
        sorted_idx ``[N]`` int64,
        valid_signal_i ``[L]`` uint32 (data-dependent *L*),
        valid_signal_o ``[L]`` uint32,
        valid_signal_seg ``[V+1]`` uint32.
    """
    return _C.neighbor_map_post_process_for_masked_implicit_gemm_1(
        neighbor_map
    )


@neighbor_map_post_process_for_masked_implicit_gemm_1.register_fake
def _(
    neighbor_map: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    N = neighbor_map.shape[0]
    V = neighbor_map.shape[1]
    ctx = torch.library.get_ctx()
    L = ctx.new_dynamic_size()
    dev = neighbor_map.device
    return (
        torch.empty(N, dtype=torch.int32, device=dev),
        torch.empty(N, dtype=torch.int64, device=dev),
        torch.empty(L, dtype=torch.uint32, device=dev),
        torch.empty(L, dtype=torch.uint32, device=dev),
        torch.empty(V + 1, dtype=torch.uint32, device=dev),
    )


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::neighbor_map_post_process_for_masked_implicit_gemm_2",
    mutates_args=(),
)
def neighbor_map_post_process_for_masked_implicit_gemm_2(
    gray_code: Tensor,
    sorted_idx: Tensor,
    block_size: int,
) -> Tuple[Tensor, Tensor]:
    """Compute valid kernel indices for masked implicit GEMM.

    Returns:
        valid_kernel_idx ``[L]`` int32 (data-dependent *L*),
        seglen ``[num_blocks+1]`` int32.
    """
    return _C.neighbor_map_post_process_for_masked_implicit_gemm_2(
        gray_code, sorted_idx, block_size
    )


@neighbor_map_post_process_for_masked_implicit_gemm_2.register_fake
def _(
    gray_code: Tensor,
    sorted_idx: Tensor,
    block_size: int,
) -> Tuple[Tensor, Tensor]:
    N = gray_code.shape[0]
    num_blocks = (N + block_size - 1) // block_size
    ctx = torch.library.get_ctx()
    L = ctx.new_dynamic_size()
    dev = gray_code.device
    return (
        torch.empty(L, dtype=torch.int32, device=dev),
        torch.empty(num_blocks + 1, dtype=torch.int32, device=dev),
    )
