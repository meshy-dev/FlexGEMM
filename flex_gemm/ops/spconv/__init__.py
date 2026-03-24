import os

class Algorithm:
    """Algorithm choices for sparse convolution."""
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"


ALGORITHM = Algorithm.MASKED_IMPLICIT_GEMM_SPLITK  # Default algorithm
HASHMAP_RATIO = 2.0         # Ratio of hashmap size to input size

# How to build the submanifold neighbor map:
# - "hashmap": CUDA hashmap (default; fastest on GPU, requires CUDA extension).
# - "searchsorted": CUDA Morton encode + sort + torch.searchsorted on GPU (CPU uses
#   pure torch for encode). Works on CPU for explicit/implicit neighbor maps only.
#   Set ``FLEX_GEMM_SPATIAL_INDEX_MODE=searchsorted`` before ``import flex_gemm`` to
#   match this default at import time (JIT CUDA is then skipped until a native op runs).
_env_spatial = os.environ.get("FLEX_GEMM_SPATIAL_INDEX_MODE", "").lower()
if _env_spatial in ("hashmap", "searchsorted"):
    SPATIAL_INDEX_MODE = _env_spatial
else:
    SPATIAL_INDEX_MODE = "hashmap"
# Bit shift on spatial coords before Morton (same semantics as meshy_sparse morton3d_16).
SEARCHSORTED_SKIP_K = 0


def set_algorithm(algorithm: Algorithm):
    global ALGORITHM
    ALGORITHM = algorithm


def set_hashmap_ratio(ratio: float):
    global HASHMAP_RATIO
    HASHMAP_RATIO = ratio


def set_spatial_index_mode(mode: str) -> None:
    """Set neighbor-map construction strategy: ``\"hashmap\"`` or ``\"searchsorted\"``."""
    global SPATIAL_INDEX_MODE
    if mode not in ("hashmap", "searchsorted"):
        raise ValueError(f"mode must be 'hashmap' or 'searchsorted', got {mode!r}")
    SPATIAL_INDEX_MODE = mode


def set_searchsorted_skip_k(skip_k: int) -> None:
    """Morton ``skip_k`` for ``searchsorted`` mode (must be in ``[0, 15]``)."""
    global SEARCHSORTED_SKIP_K
    if not isinstance(skip_k, int) or not (0 <= skip_k <= 15):
        raise ValueError(f"skip_k must be int in [0, 15], got {skip_k!r}")
    SEARCHSORTED_SKIP_K = skip_k


from .submanifold_conv3d import (
    SubMConv3dFunction,
    SubMConv3dNeighborCache,
    SpConvConfig,
    sparse_submanifold_conv3d,
)
from . import _custom_ops as _spconv_custom_ops  # noqa: F401 — registers custom ops
