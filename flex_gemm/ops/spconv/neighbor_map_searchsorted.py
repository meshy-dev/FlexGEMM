"""
Build submanifold sparse-convolution neighbor maps via sorted Morton keys + searchsorted.

Same pattern as ``meshy_sparse.sparse_sample.sample_sparse_morton_search``: encode voxel
coordinates with **batched 48+16 Morton** (``meshy_sparse.morton3d_16`` semantics), sort,
then for each neighbor coordinate use :func:`torch.searchsorted` plus equality check.

Output layout matches ``hashmap_build_submanifold_conv_neighbour_map_cuda``: ``[N, V]``
``uint32``, ``0xFFFFFFFF`` for missing neighbors.
"""

from __future__ import annotations

import torch

from .morton_key import morton_keys_batched_ncwhd


def build_submanifold_neighbor_map_searchsorted(
    coords: torch.Tensor,
    shape: torch.Size,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
    *,
    skip_k: int = 0,
) -> torch.Tensor:
    """Return neighbor map ``[L, V]`` uint32 (invalid = ``0xFFFFFFFF``).

    Args:
        coords: ``[L, 4]`` int32, columns ``(batch, w, h, d)`` matching NCWHD ``shape``.
        shape: ``torch.Size([N, C, W, H, D])`` batch/spatial bounds.
        kernel_size: ``(Kw, Kh, Kd)`` in the same axis order as FlexGEMM conv weights.
        dilation: ``(Dw, Dh, Dd)``.
        skip_k: Bit shift on spatial coords before Morton encode (same as ``morton3d_16``).
    """
    assert coords.ndim == 2 and coords.shape[1] == 4, "coords must be [L, 4]"
    assert coords.dtype == torch.int32, "coords must be int32"
    assert 0 <= skip_k <= 15, "skip_k must be in [0, 15] (morton3d_16 range)"

    _n_b, _c, W, H, D = shape
    L = coords.shape[0]
    kw, kh, kd = kernel_size
    dw, dh, dd = dilation
    V = kw * kh * kd

    device = coords.device

    keys = morton_keys_batched_ncwhd(coords, skip_k=skip_k)
    sorted_keys, sort_idx = torch.sort(keys)

    dz = torch.arange(
        -(kw // 2) * dw,
        kw // 2 * dw + 1,
        dw,
        device=device,
        dtype=torch.int32,
    )
    dy = torch.arange(
        -(kh // 2) * dh,
        kh // 2 * dh + 1,
        dh,
        device=device,
        dtype=torch.int32,
    )
    dx = torch.arange(
        -(kd // 2) * dd,
        kd // 2 * dd + 1,
        dd,
        device=device,
        dtype=torch.int32,
    )
    gz, gy, gx = torch.meshgrid(dz, dy, dx, indexing="ij")
    offset = torch.stack([gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)], dim=-1)

    neighbor_coords = coords.unsqueeze(1).expand(L, V, 4).clone()
    neighbor_coords[:, :, 1:4] = neighbor_coords[:, :, 1:4] + offset.unsqueeze(0).to(
        torch.int32
    )
    neighbor_coords_flat = neighbor_coords.reshape(L * V, 4)

    neighbor_valid = (
        (neighbor_coords_flat[:, 1] >= 0)
        & (neighbor_coords_flat[:, 1] < W)
        & (neighbor_coords_flat[:, 2] >= 0)
        & (neighbor_coords_flat[:, 2] < H)
        & (neighbor_coords_flat[:, 3] >= 0)
        & (neighbor_coords_flat[:, 3] < D)
    )

    neighbor_keys = morton_keys_batched_ncwhd(neighbor_coords_flat, skip_k=skip_k)

    pos = torch.searchsorted(sorted_keys, neighbor_keys)
    pos = pos.clamp_(0, sorted_keys.numel() - 1)
    hit = neighbor_valid & (sorted_keys[pos] == neighbor_keys)

    out_flat = torch.full(
        (L * V,),
        0xFFFFFFFF,
        dtype=torch.long,
        device=device,
    )
    out_flat[hit] = sort_idx[pos[hit]].to(torch.long)

    return out_flat.view(L, V).to(torch.uint32)
