# FlexGEMM

CUDA/Triton kernels for submanifold sparse 3D convolution. Installed as the `flex_gemm` Python package. Used by `meshy_sparse/explicit_conv.py` as the accelerated backend.

## Submodule Status

This is a **git submodule** inside `packages/meshy_sparse/third_party/FlexGEMM/`.

- **Current branch**: `dev/custom_ops` (not `main`)
- **Upstream**: `JeffreyXiang/FlexGEMM` on GitHub
- The workspace `meshylatv_sparse_sdf_dev` installs from this local submodule
- The workspace `meshy_sparse_dev` installs from GitHub upstream (not this local copy)

## This Is a Package, Not a Workspace

FlexGEMM has no `pixi.toml`. Run all commands from a workspace that has `flex_gemm` installed:

```bash
cd workspace/meshylatv_sparse_sdf_dev
# then use pixi run/shell from there
```

## First Import: JIT Compilation

The first `import flex_gemm` triggers JIT CUDA extension compilation via `torch.utils.cpp_extension`. This can take **several minutes**. Subsequent imports use the cache at `~/.cache/torch_extensions/`.

If you change CUDA source files and need to force recompilation:
```bash
rm -rf ~/.cache/torch_extensions/
```

## Key APIs

```python
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, SpConvConfig, Algorithm, set_algorithm
from flex_gemm.ops.grid_sample import grid_sample_3d
```

Two calling conventions for sparse conv:

```python
# Legacy (builds neighbor cache, not torch.compile-friendly)
out, neighbor_cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

# Compiled (torch.compile-friendly, requires frozen config)
config = neighbor_cache.freeze(splitk=1, algorithm="masked_implicit_gemm_splitk")
out = sparse_submanifold_conv3d(feats, weight=weight, bias=bias, config=config)
```

## Custom Op Categories

| Category | `torch.ops.flex_gemm.*` ops |
|---|---|
| Hashmap | `hashmap_insert_cuda`, `hashmap_lookup_cuda`, `hashmap_insert_3d_cuda`, `hashmap_lookup_3d_cuda`, `hashmap_insert_3d_idx_as_val_cuda` |
| Serialize | `z_order_encode`, `z_order_decode`, `hilbert_encode`, `hilbert_decode` |
| Grid sample | `hashmap_build_grid_sample_3d_nearest_neighbor_map`, `hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight` |
| Spconv neighbor map | `hashmap_build_submanifold_conv_neighbour_map_cuda`, `neighbor_map_post_process_for_masked_implicit_gemm_1`, `neighbor_map_post_process_for_masked_implicit_gemm_2` |
| Spconv kernels | `sparse_conv_masked_fwd`, `sparse_conv_masked_bwd`, `sparse_conv_masked_splitk_fwd`, `sparse_conv_masked_splitk_bwd` |

## Running Tests

All tests require a GPU and are **plain Python scripts** (not pytest). Do not run them with `pytest`.

```bash
# Get a GPU node first
srun -G 1 --pty bash

# cd into a workspace that has flex_gemm installed
cd workspace/meshylatv_sparse_sdf_dev

# Full custom-ops test suite
pixi run -e default python ../../packages/meshy_sparse/third_party/FlexGEMM/tests/custom_ops/run_all.py

# Individual test files
pixi run -e default python ../../packages/meshy_sparse/third_party/FlexGEMM/tests/custom_ops/test_spconv.py
pixi run -e default python ../../packages/meshy_sparse/third_party/FlexGEMM/tests/custom_ops/test_grid_sample.py
pixi run -e default python ../../packages/meshy_sparse/third_party/FlexGEMM/tests/custom_ops/test_hashmap.py
pixi run -e default python ../../packages/meshy_sparse/third_party/FlexGEMM/tests/custom_ops/test_serialize.py
pixi run -e default python ../../packages/meshy_sparse/third_party/FlexGEMM/tests/custom_ops/test_neighbor_cache.py
```

`run_all.py` invokes each test file as a subprocess. Individual test files use a `main() -> int` pattern with `print`-based pass/fail reporting:

```python
def test_my_op() -> bool:
    out = torch.ops.flex_gemm.my_op(...)
    ref = reference_implementation(...)
    ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
    print(f"  max diff = {diff:.6e}  {PASS if ok else FAIL}")
    return ok

def main() -> int:
    results = {"my_op": test_my_op()}
    return print_summary(results)  # from utils.py

if __name__ == "__main__":
    sys.exit(main())
```

Shared helpers in `tests/custom_ops/utils.py`: `sphere_coords()`, `run_test()` (torch.compile graph-break detection), `print_summary()`.

## torch.compile Notes

The `dev/custom_ops` branch adds full `torch.compile` support. Key design decisions (detailed in `docs/custom_ops.md`):

- All CUDA pybind11 functions wrapped with `@torch.library.custom_op` and `register_fake`
- Triton launchers split into `_impl()` + custom_op wrapper to avoid nested dispatch
- `SubMConv3dNeighborCache.freeze()` converts Python neighbor-cache into `SpConvConfig` (pure tensors) capturable by `torch.compile`
- Do **not** call custom_op A from inside custom_op B's implementation -- use `_impl` directly

## Building

FlexGEMM requires no-build-isolation (depends on already-installed PyTorch):

```toml
# Already configured in workspace pixi.toml
[pypi-options]
no-build-isolation = ["flex-gemm"]
```

## Submodule Workflow

```bash
# Enter the submodule
cd packages/meshy_sparse/third_party/FlexGEMM

# Make changes and commit inside the submodule
git add <files>
git commit -m "your message"

# Then update the parent repo's submodule pointer (from repo root)
cd ../../../..
git add packages/meshy_sparse/third_party/FlexGEMM
git commit -m "update FlexGEMM submodule"
```
