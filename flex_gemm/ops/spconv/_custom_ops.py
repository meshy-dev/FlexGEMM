"""
Custom op wrappers for masked sparse submanifold convolution.

These ops accept pre-computed ``valid_kernel`` / ``valid_kernel_seg``
tensors (for every possible autotune block-size) instead of Python
callbacks, making them compatible with ``torch.compile``.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ... import kernels
from ...kernels.triton.spconv import (
    sparse_submanifold_conv_fwd_masked_implicit_gemm,
    sparse_submanifold_conv_bwd_masked_implicit_gemm,
    sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk,
    sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk,
)


# ============================================================
# Helpers
# ============================================================


def _build_callback(
    block_sizes: Sequence[int],
    tensors: Sequence[Tensor],
):
    """Return a callback ``f(block_size) -> Tensor`` backed by a dict."""
    mapping = dict(zip(block_sizes, tensors))
    return lambda bs: mapping[bs]


# ============================================================
# MASKED_IMPLICIT_GEMM  (non-split-K)
# ============================================================


@torch.library.custom_op(
    "flex_gemm::sparse_conv_masked_fwd", mutates_args=()
)
def sparse_conv_masked_fwd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    neighbor: Tensor,
    sorted_idx: Tensor,
    valid_kernels: Sequence[Tensor],
    valid_kernel_segs: Sequence[Tensor],
    block_sizes: Sequence[int],
) -> Tensor:
    """Masked implicit-GEMM forward (non-split-K)."""
    return sparse_submanifold_conv_fwd_masked_implicit_gemm(
        input,
        weight,
        bias,
        neighbor,
        sorted_idx,
        valid_kernel=_build_callback(block_sizes, valid_kernels),
        valid_kernel_seg=_build_callback(block_sizes, valid_kernel_segs),
    )


@sparse_conv_masked_fwd.register_fake
def _(input, weight, bias, neighbor, sorted_idx, valid_kernels, valid_kernel_segs, block_sizes):
    N = neighbor.shape[0]
    Co = weight.shape[0]
    return input.new_empty(N, Co)


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::sparse_conv_masked_bwd", mutates_args=()
)
def sparse_conv_masked_bwd(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    neighbor: Tensor,
    sorted_idx: Tensor,
    valid_kernels: Sequence[Tensor],
    valid_kernel_segs: Sequence[Tensor],
    block_sizes: Sequence[int],
    valid_signal_i: Tensor,
    valid_signal_o: Tensor,
    valid_signal_seg: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Masked implicit-GEMM backward (non-split-K).

    Always computes all three gradients (input, weight, bias).

    Inside a ``custom_op``, autograd is disabled so
    ``tensor.requires_grad`` is ``False``.  The underlying backward
    function uses ``requires_grad`` to skip unused gradient
    computations, so we force-enable it here.
    """
    # Force requires_grad so the inner function computes all gradients.
    input = input.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    if bias is not None:
        bias = bias.detach().requires_grad_(True)

    grad_input, grad_weight, grad_bias = (
        sparse_submanifold_conv_bwd_masked_implicit_gemm(
            grad_output.contiguous(),
            input,
            weight,
            bias,
            neighbor,
            sorted_idx,
            valid_kernel=_build_callback(block_sizes, valid_kernels),
            valid_kernel_seg=_build_callback(block_sizes, valid_kernel_segs),
            valid_signal_i=valid_signal_i,
            valid_signal_o=valid_signal_o,
            valid_signal_seg=valid_signal_seg,
        )
    )
    N, Ci = input.shape
    Co, V = weight.shape[0], weight.shape[1]
    if grad_input is None:
        grad_input = torch.zeros(N, Ci, device=input.device, dtype=input.dtype)
    if grad_weight is None:
        grad_weight = torch.zeros(
            Co, V, Ci, device=weight.device, dtype=weight.dtype
        )
    if grad_bias is None:
        grad_bias = torch.zeros(Co, device=input.device, dtype=input.dtype)
    return grad_input, grad_weight, grad_bias


@sparse_conv_masked_bwd.register_fake
def _(
    grad_output, input, weight, bias, neighbor, sorted_idx,
    valid_kernels, valid_kernel_segs, block_sizes,
    valid_signal_i, valid_signal_o, valid_signal_seg,
):
    N, Ci = input.shape
    Co, V = weight.shape[0], weight.shape[1]
    return (
        input.new_empty(N, Ci),
        weight.new_empty(Co, V, Ci),
        input.new_empty(Co),
    )


# ============================================================
# MASKED_IMPLICIT_GEMM_SPLITK
# ============================================================


@torch.library.custom_op(
    "flex_gemm::sparse_conv_masked_splitk_fwd", mutates_args=()
)
def sparse_conv_masked_splitk_fwd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    neighbor: Tensor,
    sorted_idx: Tensor,
    valid_kernels: Sequence[Tensor],
    valid_kernel_segs: Sequence[Tensor],
    block_sizes: Sequence[int],
    splitk: int,
) -> Tensor:
    """Masked implicit-GEMM split-K forward.

    Bypasses the ``@autotune`` wrapper and calls the underlying kernel
    function directly with a fixed SPLITK value (chosen during
    ``freeze()``).  This avoids the autotune ``key_fn`` rejecting the
    extra ``SPLITK`` kwarg.
    """
    # The imported name is a PersistentCacheAutoTuner instance.
    # Access .kernel to get the raw function that accepts SPLITK.
    _raw_fn = sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.kernel
    return _raw_fn(
        input,
        weight,
        bias,
        neighbor,
        sorted_idx,
        valid_kernel=_build_callback(block_sizes, valid_kernels),
        valid_kernel_seg=_build_callback(block_sizes, valid_kernel_segs),
        SPLITK=splitk,
    )


@sparse_conv_masked_splitk_fwd.register_fake
def _(input, weight, bias, neighbor, sorted_idx, valid_kernels, valid_kernel_segs, block_sizes, splitk):
    N = neighbor.shape[0]
    Co = weight.shape[0]
    return input.new_empty(N, Co)


# ------------------------------------------------------------------


@torch.library.custom_op(
    "flex_gemm::sparse_conv_masked_splitk_bwd", mutates_args=()
)
def sparse_conv_masked_splitk_bwd(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    neighbor: Tensor,
    sorted_idx: Tensor,
    valid_kernels: Sequence[Tensor],
    valid_kernel_segs: Sequence[Tensor],
    block_sizes: Sequence[int],
    valid_signal_i: Tensor,
    valid_signal_o: Tensor,
    valid_signal_seg: Tensor,
    splitk: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Masked implicit-GEMM split-K backward.

    Always computes all three gradients (input, weight, bias).

    See :func:`sparse_conv_masked_bwd` for the ``requires_grad``
    rationale.
    """
    # Force requires_grad so the inner function computes all gradients.
    input = input.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)
    if bias is not None:
        bias = bias.detach().requires_grad_(True)

    grad_input, grad_weight, grad_bias = (
        sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
            grad_output.contiguous(),
            input,
            weight,
            bias,
            neighbor,
            sorted_idx,
            valid_kernel=_build_callback(block_sizes, valid_kernels),
            valid_kernel_seg=_build_callback(block_sizes, valid_kernel_segs),
            valid_signal_i=valid_signal_i,
            valid_signal_o=valid_signal_o,
            valid_signal_seg=valid_signal_seg,
        )
    )
    N, Ci = input.shape
    Co, V = weight.shape[0], weight.shape[1]
    if grad_input is None:
        grad_input = torch.zeros(N, Ci, device=input.device, dtype=input.dtype)
    if grad_weight is None:
        grad_weight = torch.zeros(
            Co, V, Ci, device=weight.device, dtype=weight.dtype
        )
    if grad_bias is None:
        grad_bias = torch.zeros(Co, device=input.device, dtype=input.dtype)
    return grad_input, grad_weight, grad_bias


@sparse_conv_masked_splitk_bwd.register_fake
def _(
    grad_output, input, weight, bias, neighbor, sorted_idx,
    valid_kernels, valid_kernel_segs, block_sizes,
    valid_signal_i, valid_signal_o, valid_signal_seg,
    splitk,
):
    N, Ci = input.shape
    Co, V = weight.shape[0], weight.shape[1]
    return (
        input.new_empty(N, Ci),
        weight.new_empty(Co, V, Ci),
        input.new_empty(Co),
    )
