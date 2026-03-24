# FlexGEMM Custom Ops: `torch.compile` Support

## Motivation

FlexGEMM's CUDA extensions (hashmap, spconv neighbor-map) and
Triton kernel launchers were originally exposed via pybind11 or plain
Python functions. `torch.compile` cannot trace through either of these,
causing **graph breaks** at every kernel call.

This document describes the interface changes that make FlexGEMM fully
compatible with `torch.compile`.

---

## Architecture Overview

```
                     ┌─────────────────────────────────┐
                     │       User Code                 │
                     │  sparse_submanifold_conv3d(...)  │
                     └──────────┬──────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │ config= provided?           │
           ┌─────┴─────┐                ┌──────┴──────┐
           │ No (legacy)│                │ Yes (compile)│
           │ SubMConv3d │                │ SubMConv3d  │
           │ Function   │                │ Compiled    │
           │ (unchanged)│                │ Function    │
           └─────┬──────┘                └──────┬──────┘
                 │                              │
        Python objects,               torch.ops.flex_gemm.*
        callbacks, globals            (custom_ops, pure tensors)
                 │                              │
           ┌─────┴──────┐              ┌────────┴────────┐
           │ pybind11   │              │ register_fake   │
           │ CUDA ext   │              │ (FakeTensor for │
           │ + Triton   │              │  torch.compile) │
           └────────────┘              └─────────────────┘
```

---

## What Changed

### Neighbor-map build mode (`searchsorted` vs hashmap)

**Module**: `flex_gemm/ops/spconv/__init__.py`

- ``spconv.set_spatial_index_mode("hashmap")`` (default): build ``neighbor_map`` with the CUDA hashmap path (fast on GPU).
- ``spconv.set_spatial_index_mode("searchsorted")``: build the same ``[N, V]`` map with sorted **batched 48+16 Morton keys** (same encoding as ``meshy_sparse.morton3d_16`` on ``[L, 4]`` coords) + :func:`torch.searchsorted`. On CUDA, Morton encoding uses pybind kernels ``morton_keys_batched_ncwhd_cuda`` / ``morton_keys_3d_16_cuda``; on CPU, use the PyTorch references in ``morton_key.py``. Use ``set_searchsorted_skip_k(k)`` for ``skip_k`` (``0..15``). Masked post-process still uses CUDA.

### Layer 1: CUDA Extension Ops

**File**: `flex_gemm/kernels/_cuda_custom_ops.py` (new)

The underlying pybind module is JIT-built via `flex_gemm/kernels/_cuda_jit.py`
(`torch.utils.cpp_extension.load`), not a build-time `CUDAExtension`. Python
access goes through `flex_gemm/kernels/cuda.py`, which loads the extension on
first attribute use.

Two additional pybind entry points (Morton only) are exposed directly on the
JIT module (``kernels.cuda.morton_keys_*_cuda``), not as ``torch.ops.flex_gemm``:

``morton_keys_3d_16_cuda``, ``morton_keys_batched_ncwhd_cuda``.

All 8 spconv/hashmap pybind11 CUDA functions are wrapped with `@torch.library.custom_op`
and `register_fake`:

| Category | Ops | `mutates_args` |
|----------|-----|----------------|
| Hashmap | `hashmap_insert_cuda`, `hashmap_lookup_cuda`, `hashmap_insert_3d_cuda`, `hashmap_lookup_3d_cuda`, `hashmap_insert_3d_idx_as_val_cuda` | insert ops mutate hashmap |
| Spconv | `hashmap_build_submanifold_conv_neighbour_map_cuda`, `neighbor_map_post_process_for_masked_implicit_gemm_1`, `neighbor_map_post_process_for_masked_implicit_gemm_2` | mutate hashmap |

**Key detail**: Ops with data-dependent output sizes (post-process ops) use
`torch.library.get_ctx().new_dynamic_size()` for unbacked SymInts.

### Layer 2: Sparse Convolution (Compiled Path)

**Files**: `flex_gemm/ops/spconv/submanifold_conv3d.py`, `flex_gemm/ops/spconv/_custom_ops.py`

This is the biggest change. The obstacles were:

| Problem | Solution |
|---------|----------|
| `SubMConv3dNeighborCache` (Python object) | New `SpConvConfig` dataclass (tensors + ints only) |
| `valid_kernel_callback: Callable[[int], Tensor]` | Pre-compute for all block sizes in `freeze()`, store as `List[Tensor]` |
| `spconv.ALGORITHM` global dispatch | `algorithm: str` field in `SpConvConfig` |
| `@autotune` SPLITK selection | `splitk: int` field in `SpConvConfig`; call `.kernel` directly to bypass autotune |
| `requires_grad` False inside custom_op | `detach().requires_grad_(True)` before calling inner bwd function |

#### New types

```python
@dataclasses.dataclass
class SpConvConfig:
    neighbor_map: Tensor          # [N, V]
    sorted_idx: Tensor            # [N]
    valid_kernels: List[Tensor]   # per-block-size valid kernel indices
    valid_kernel_segs: List[Tensor]
    block_sizes: List[int]        # [32, 64, 128, 256]
    valid_signal_i: Tensor
    valid_signal_o: Tensor
    valid_signal_seg: Tensor
    splitk: int = 1
    algorithm: str = "masked_implicit_gemm_splitk"
```

#### New methods

- `SubMConv3dNeighborCache.freeze(splitk, algorithm) -> SpConvConfig`

#### New custom ops (4 total)

| Op | Purpose |
|----|---------|
| `flex_gemm::sparse_conv_masked_fwd` | Masked GEMM forward (non-split-K) |
| `flex_gemm::sparse_conv_masked_bwd` | Masked GEMM backward |
| `flex_gemm::sparse_conv_masked_splitk_fwd` | Masked GEMM split-K forward |
| `flex_gemm::sparse_conv_masked_splitk_bwd` | Masked GEMM split-K backward |

#### Updated public API

```python
# Legacy (unchanged):
out, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

# Compiled (new):
out = sparse_submanifold_conv3d(feats, weight=weight, bias=bias, config=config)
```

### Call-site Migration

All `kernels.cuda.xxx(...)` calls in `ops/` files were switched to
`torch.ops.flex_gemm.xxx(...)`, **except** inside custom_op real
implementations (to avoid nested dispatch).

| File | Calls changed |
|------|---------------|
| `ops/spconv/submanifold_conv3d.py` | 4 |

---

## Lessons Learned

1. **Nested custom_op dispatch**: Custom_op A calling custom_op B can cause
   B to dispatch to `register_fake` (FakeTensor) during compiled execution,
   producing garbage data. Fix: extract `_impl` functions and call directly.

2. **`requires_grad` inside custom_ops**: `@torch.library.custom_op`
   implementations run with autograd disabled, so all tensors appear to have
   `requires_grad=False`. If the inner function checks this flag to skip
   gradient computation, call `.detach().requires_grad_(True)` first.

3. **`@autotune` + custom_ops**: The `PersistentCacheAutoTuner.__call__`
   forwards all kwargs to `key_fn`, which rejects unknown kwargs like
   `SPLITK`. Fix: bypass autotune by calling `.kernel` (the raw function)
   directly when SPLITK is already known.

4. **`setup_context` + `mark_non_differentiable`**: The new-style
   `autograd.Function` with `setup_context` and `mark_non_differentiable`
   has interaction issues with `torch.compile`. Fix: use old-style
   `forward(ctx, ...)` that returns a single Tensor.
