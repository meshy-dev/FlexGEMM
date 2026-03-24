"""Lazy access to the JIT-compiled CUDA/pybind extension.

The shared library is built on first attribute access (not at ``import flex_gemm``),
so workloads that use ``SPATIAL_INDEX_MODE='searchsorted'`` with explicit/implicit
algorithms never trigger a compile. Native code is still required for hashmap
neighbor maps and masked post-process.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    from ._cuda_jit import load_cuda_extension

    return getattr(load_cuda_extension(), name)
