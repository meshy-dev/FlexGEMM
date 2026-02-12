#!/usr/bin/env python
"""
Minimal example: compiled sparse submanifold convolution (fwd + bwd).

Demonstrates the new ``SpConvConfig``-based API that is compatible with
``torch.compile``.  The workflow is:

    1. Create sparse data (sphere shell).
    2. Build the neighbor cache directly from geometry (coords, shape,
       kernel_size, dilation) -- no forward pass needed.
    3. Freeze the cache into a ``SpConvConfig`` (pure tensors, no callbacks).
    4. Use ``sparse_submanifold_conv3d(..., config=config)`` for all
       subsequent forward/backward calls -- this path is fully
       ``torch.compile``-friendly.

Usage::

    srun -G 1 python tests/custom_ops/example_spconv.py
"""

import torch
from utils import sphere_coords

from flex_gemm.ops.spconv import (
    Algorithm,
    set_algorithm,
    sparse_submanifold_conv3d,
)
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction

# -----------------------------------------------------------------
# 1. Create sparse data
# -----------------------------------------------------------------
RES, CH = 64, 128
feats, coords, shape = sphere_coords(RES, CH, dtype=torch.float32)
weight = torch.randn(CH, 3, 3, 3, CH, device="cuda")
bias = torch.randn(CH, device="cuda")

print(f"Resolution : {RES}")
print(f"Channels   : {CH}")
print(f"N (voxels) : {feats.shape[0]}")
print(f"Shape      : {shape}")
print()

# -----------------------------------------------------------------
# 2. Build neighbor cache from geometry (no forward pass needed)
# -----------------------------------------------------------------
KERNEL_SIZE = (3, 3, 3)
DILATION = (1, 1, 1)

set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
    coords, shape, KERNEL_SIZE, DILATION,
)

# -----------------------------------------------------------------
# 3. Freeze into a compile-friendly SpConvConfig
# -----------------------------------------------------------------
config = neighbor_cache.freeze(splitk=1, algorithm="masked_implicit_gemm_splitk")

print(f"SpConvConfig:")
print(f"  neighbor_map   : {config.neighbor_map.shape}")
print(f"  sorted_idx     : {config.sorted_idx.shape}")
print(f"  block_sizes    : {config.block_sizes}")
print(f"  valid_kernels  : {[v.shape for v in config.valid_kernels]}")
print(f"  algorithm      : {config.algorithm}")
print(f"  splitk         : {config.splitk}")
print()

# -----------------------------------------------------------------
# 4. Forward pass (compiled path)
# -----------------------------------------------------------------
feats = feats.detach().requires_grad_(True)
weight = weight.detach().requires_grad_(True)
bias = bias.detach().requires_grad_(True)

out = sparse_submanifold_conv3d(feats, weight=weight, bias=bias, config=config)
print(f"Forward output   : {out.shape}")

# -----------------------------------------------------------------
# 5. Backward pass
# -----------------------------------------------------------------
loss = out.sum()
loss.backward()

print(f"grad_feats       : {feats.grad.shape}")
print(f"grad_weight      : {weight.grad.shape}")
print(f"grad_bias        : {bias.grad.shape}")
print()

# -----------------------------------------------------------------
# 6. With torch.compile (fullgraph=True, no graph breaks)
# -----------------------------------------------------------------
feats_c = feats.detach().requires_grad_(True)
weight_c = weight.detach().requires_grad_(True)
bias_c = bias.detach().requires_grad_(True)


@torch.compile(fullgraph=True)
def compiled_conv(feats, weight, bias):
    return sparse_submanifold_conv3d(feats, weight=weight, bias=bias, config=config)


out_c = compiled_conv(feats_c, weight_c, bias_c)
out_c.sum().backward()

# Verify compiled vs eager
fwd_diff = (out.detach().float() - out_c.detach().float()).abs().max().item()
bwd_diff = (feats.grad.float() - feats_c.grad.float()).abs().max().item()
print(f"torch.compile fullgraph=True:")
print(f"  fwd max diff   : {fwd_diff:.6e}")
print(f"  bwd max diff   : {bwd_diff:.6e}")
print()

# -----------------------------------------------------------------
# 7. Quick benchmark: eager vs compiled
# -----------------------------------------------------------------
import time

torch.cuda.synchronize()

# Warmup
for _ in range(5):
    compiled_conv(feats_c, weight_c, bias_c)
    sparse_submanifold_conv3d(feats, weight=weight, bias=bias, config=config)
torch.cuda.synchronize()

N_ITER = 50

start = time.time()
for _ in range(N_ITER):
    sparse_submanifold_conv3d(feats, weight=weight, bias=bias, config=config)
torch.cuda.synchronize()
eager_ms = (time.time() - start) / N_ITER * 1000

start = time.time()
for _ in range(N_ITER):
    compiled_conv(feats_c, weight_c, bias_c)
torch.cuda.synchronize()
compiled_ms = (time.time() - start) / N_ITER * 1000

print(f"Benchmark ({N_ITER} iters):")
print(f"  Eager    : {eager_ms:.3f} ms")
print(f"  Compiled : {compiled_ms:.3f} ms")
print(f"  Speedup  : {eager_ms / compiled_ms:.2f}x")
