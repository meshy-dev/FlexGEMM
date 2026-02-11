from typing import *

import torch
from torch import Tensor
from torch.autograd import Function

from .. import grid_sample, utils
from ... import kernels
from ...kernels.triton.grid_sample.indice_weighed_sum_fwd import (
    _indice_weighed_sum_fwd_impl,
)
from ...kernels.triton.grid_sample.indice_weighed_sum_bwd import (
    _indice_weighed_sum_bwd_input_impl,
)


# ============================================================
# Custom ops for grid_sample_3d (torch.compile compatible)
# ============================================================


@torch.library.custom_op("flex_gemm::_grid_sample_3d_nearest_fwd", mutates_args=())
def _grid_sample_3d_nearest_fwd(
    feats: Tensor,
    coords: Tensor,
    query_pts: Tensor,
    shape: Sequence[int],
    hashmap_ratio: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Nearest-neighbor grid sampling forward pass.

    Args:
        feats: [N, C] input features
        coords: [N, 4] coordinates
        query_pts: [B, L, 3] query points
        shape: spatial shape (N_batch, C, W, H, D)
        hashmap_ratio: ratio of hashmap size to input size

    Returns:
        output: [B, L, C] sampled features
        indices: [B, L] int32 neighbor indices (clamped to >= 0)
        valid: [B, L] bool validity mask
    """
    assert feats.dim() == 2, f"Features must be of shape [N, C], got {feats.shape}"
    assert (
        coords.dim() == 2 and coords.shape[1] == 4
    ), f"Coords must be of shape [N, 4], got {coords.shape}"
    assert (
        query_pts.dim() == 3 and query_pts.shape[2] == 3
    ), f"Query points must be of shape [B, L, 3], got {query_pts.shape}"
    assert (
        feats.shape[0] == coords.shape[0]
    ), "Number of features must match number of coordinates"

    N = coords.shape[0]
    B, L = query_pts.shape[:2]
    C, W, H, D = shape[-4:]

    hashmap_keys, hashmap_vals = utils.init_hashmap(
        shape, int(hashmap_ratio * N), coords.device
    )
    indices = kernels.cuda.hashmap_build_grid_sample_3d_nearest_neighbor_map(
        hashmap_keys,
        hashmap_vals,
        coords.int(),
        query_pts,
        W,
        H,
        D,
    ).int()
    valid = indices != 0xFFFFFFFF
    indices.clamp_min_(0)
    out = valid.unsqueeze(-1) * feats.index_select(
        0, indices.reshape(-1)
    ).reshape(B, L, C)

    return out, indices, valid


@_grid_sample_3d_nearest_fwd.register_fake
def _(
    feats: Tensor,
    coords: Tensor,
    query_pts: Tensor,
    shape: Sequence[int],
    hashmap_ratio: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    B, L = query_pts.shape[:2]
    C = feats.shape[1]
    return (
        feats.new_empty(B, L, C),
        torch.empty(B, L, dtype=torch.int32, device=feats.device),
        torch.empty(B, L, dtype=torch.bool, device=feats.device),
    )


@torch.library.custom_op(
    "flex_gemm::_grid_sample_3d_trilinear_fwd", mutates_args=()
)
def _grid_sample_3d_trilinear_fwd(
    feats: Tensor,
    coords: Tensor,
    query_pts: Tensor,
    shape: Sequence[int],
    hashmap_ratio: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Trilinear grid sampling forward pass.

    Args:
        feats: [N, C] input features
        coords: [N, 4] coordinates
        query_pts: [B, L, 3] query points
        shape: spatial shape (N_batch, C, W, H, D)
        hashmap_ratio: ratio of hashmap size to input size

    Returns:
        output: [B, L, C] sampled features
        indices: [B*L, 8] neighbor indices
        interp_weight: [B*L, 8] interpolation weights
    """
    assert feats.dim() == 2, f"Features must be of shape [N, C], got {feats.shape}"
    assert (
        coords.dim() == 2 and coords.shape[1] == 4
    ), f"Coords must be of shape [N, 4], got {coords.shape}"
    assert (
        query_pts.dim() == 3 and query_pts.shape[2] == 3
    ), f"Query points must be of shape [B, L, 3], got {query_pts.shape}"
    assert (
        feats.shape[0] == coords.shape[0]
    ), "Number of features must match number of coordinates"

    N = coords.shape[0]
    B, L = query_pts.shape[:2]
    C, W, H, D = shape[-4:]

    hashmap_keys, hashmap_vals = utils.init_hashmap(
        shape, int(hashmap_ratio * N), coords.device
    )
    indices, interp_weight = (
        kernels.cuda.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
            hashmap_keys,
            hashmap_vals,
            coords.int(),
            query_pts,
            W,
            H,
            D,
        )
    )

    out = _indice_weighed_sum_fwd_impl(
        feats,
        indices.view(-1, 8),
        interp_weight.view(-1, 8),
    ).view(B, L, C)

    return (
        out,
        indices.reshape(-1, 8).contiguous(),
        interp_weight.reshape(-1, 8).contiguous(),
    )


@_grid_sample_3d_trilinear_fwd.register_fake
def _(
    feats: Tensor,
    coords: Tensor,
    query_pts: Tensor,
    shape: Sequence[int],
    hashmap_ratio: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    B, L = query_pts.shape[:2]
    C = feats.shape[1]
    return (
        feats.new_empty(B, L, C),
        torch.empty(B * L, 8, dtype=torch.int32, device=feats.device),
        feats.new_empty(B * L, 8),
    )


@torch.library.custom_op(
    "flex_gemm::_grid_sample_3d_trilinear_bwd", mutates_args=()
)
def _grid_sample_3d_trilinear_bwd(
    grad_output: Tensor,
    indices: Tensor,
    interp_weight: Tensor,
    N: int,
) -> Tensor:
    """
    Trilinear grid sampling backward pass for input features.

    Args:
        grad_output: [M, C] gradient of the output
        indices: [M, 8] neighbor indices
        interp_weight: [M, 8] interpolation weights
        N: number of input features

    Returns:
        grad_feats: [N, C] gradient for input features
    """
    C = grad_output.shape[-1]
    return _indice_weighed_sum_bwd_input_impl(
        grad_output.contiguous(),
        indices,
        interp_weight,
        N,
    ).view(N, C)


@_grid_sample_3d_trilinear_bwd.register_fake
def _(
    grad_output: Tensor,
    indices: Tensor,
    interp_weight: Tensor,
    N: int,
) -> Tensor:
    C = grad_output.shape[-1]
    return grad_output.new_empty(N, C)


# ============================================================
# Autograd Function
# ============================================================


class GridSample3dFunction(Function):
    """
    Autograd function for sparse 3D grid sampling.

    Supports both nearest-neighbor and trilinear interpolation modes.
    Uses custom ops internally for ``torch.compile`` compatibility.
    """

    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        coords: Tensor,
        query_pts: Tensor,
        shape: Sequence[int],
        mode: str,
        hashmap_ratio: float,
    ) -> Tensor:
        ctx.mode = mode
        if mode == "nearest":
            out, indices, valid = torch.ops.flex_gemm._grid_sample_3d_nearest_fwd(
                feats, coords, query_pts, shape, hashmap_ratio
            )
            ctx.save_for_backward(indices, valid)
        else:
            out, indices, interp_weight = (
                torch.ops.flex_gemm._grid_sample_3d_trilinear_fwd(
                    feats, coords, query_pts, shape, hashmap_ratio
                )
            )
            ctx.save_for_backward(indices, interp_weight)
        ctx.N = feats.shape[0]
        ctx.C = feats.shape[1]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        indices, aux = ctx.saved_tensors

        if ctx.mode == "nearest":
            # aux is the boolean validity mask
            grad_feats = torch.zeros(
                ctx.N,
                ctx.C,
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            grad_feats.index_add_(
                0, indices[aux], grad_output[aux].reshape(-1, ctx.C)
            )
        else:
            # aux is the float interpolation weights
            grad_feats = torch.ops.flex_gemm._grid_sample_3d_trilinear_bwd(
                grad_output.reshape(-1, ctx.C).contiguous(),
                indices,
                aux,
                ctx.N,
            )

        return grad_feats, None, None, None, None, None


# ============================================================
# Public API
# ============================================================


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    grid: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    """
    Samples the input sparse tensor at the given points using the specified interpolation mode.

    Args:
        feats (torch.Tensor): A [N, C] tensor containing the features to sample from
        coords (torch.Tensor): A [N, 4] tensor containing the coordinates of the features
        shape (torch.Size): The spatial shape of the sparse tensor
        grid (torch.Tensor): A [B, L, 3] tensor containing the query points
        mode (str): The interpolation mode to use (nearest, trilinear)

    Returns:
        torch.Tensor: A [B, L, C] tensor containing the sampled features
    """
    assert mode in ["nearest", "trilinear"], "Invalid interpolation mode"

    shape_list = list(shape)
    hashmap_ratio = float(grid_sample.HASHMAP_RATIO)

    return GridSample3dFunction.apply(
        feats, coords, grid, shape_list, mode, hashmap_ratio
    )
