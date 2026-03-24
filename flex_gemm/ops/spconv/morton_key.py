"""
48-bit 3D Morton codes + batch in high bits, matching ``meshy_sparse.morton3d_16`` (batched).

Coords are ``[L, 4]`` int with columns ``(batch, W, H, D)`` (NCWHD voxel indices).
Spatial triple fed to the interleaver is ``(D, H, W)`` as ``(x, y, z)`` in the
``meshy_sparse.morton3d_16`` convention (``x = pos[:, -1]``, etc.).

On CUDA, :func:`morton_keys_batched_ncwhd` and :func:`morton_keys_3d_16` call native
kernels; pure PyTorch *reference* implementations are exposed for tests and CPU.
"""

from __future__ import annotations

import torch


def _part1by2_16(x: torch.Tensor) -> torch.Tensor:
    """Spread lower 16 bits of each integer across 48 bits (3 interleaved streams)."""
    x = x & 0x0000_FFFF
    x = (x | (x << 16)) & 0x0000_FF00_00FF
    x = (x | (x << 8)) & 0x00F0_0F00_F00F
    x = (x | (x << 4)) & 0x0C30_C30C_30C3
    x = (x | (x << 2)) & 0x1249_2492_49249
    return x


def morton_keys_3d_16_reference(coords: torch.Tensor, skip_k: int = 0) -> torch.Tensor:
    """Reference: ``[L, 3]`` int ``(W, H, D)`` -> int64 spatial Morton (48 bits), no batch term."""
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be [L, 3]"
    assert 0 <= skip_k <= 15, "skip_k must be in [0, 15]"
    pos = coords.to(torch.int64)
    wv = pos[:, 0] >> int(skip_k)
    hv = pos[:, 1] >> int(skip_k)
    dv = pos[:, 2] >> int(skip_k)
    return (
        _part1by2_16(dv)
        | (_part1by2_16(hv) << 1)
        | (_part1by2_16(wv) << 2)
    )


def morton_keys_batched_ncwhd_reference(
    coords: torch.Tensor, skip_k: int = 0
) -> torch.Tensor:
    """Reference: ``[L, 4]`` int ``(batch, W, H, D)`` -> int64 keys (batch in high bits)."""
    assert coords.ndim == 2 and coords.shape[1] == 4, "coords must be [L, 4]"
    assert 0 <= skip_k <= 15, "skip_k must be in [0, 15]"
    pos = coords.to(torch.int64)
    xv = pos[:, -1] >> int(skip_k)
    yv = pos[:, -2] >> int(skip_k)
    zv = pos[:, -3] >> int(skip_k)
    key = (
        _part1by2_16(xv)
        | (_part1by2_16(yv) << 1)
        | (_part1by2_16(zv) << 2)
    )
    return key + (pos[:, 0] << 48)


def morton_keys_3d_16(coords: torch.Tensor, skip_k: int = 0) -> torch.Tensor:
    """``[L, 3]`` int32 ``(W, H, D)`` -> int64 Morton keys (CUDA kernel if ``coords`` is CUDA)."""
    if coords.is_cuda:
        from flex_gemm.kernels import cuda as cuda_mod

        return cuda_mod.morton_keys_3d_16_cuda(coords.contiguous(), int(skip_k))
    return morton_keys_3d_16_reference(coords, skip_k=skip_k)


def morton_keys_batched_ncwhd(coords: torch.Tensor, skip_k: int = 0) -> torch.Tensor:
    """``[L, 4]`` int32; same semantics as ``meshy_sparse.morton3d_16`` (CUDA if ``coords`` is CUDA)."""
    if coords.is_cuda:
        from flex_gemm.kernels import cuda as cuda_mod

        return cuda_mod.morton_keys_batched_ncwhd_cuda(
            coords.contiguous(), int(skip_k)
        )
    return morton_keys_batched_ncwhd_reference(coords, skip_k=skip_k)
