"""JIT-compile the FlexGEMM CUDA/pybind extension with ``torch.utils.cpp_extension.load``.

Install-time ``CUDAExtension`` is intentionally not used so the built artifact matches
the currently installed PyTorch ABI. Compilation runs on first use of any native op
(e.g. hashmap neighbor build, masked post-process).
"""

from __future__ import annotations

import os
import platform
import threading
from typing import Any

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION, load

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CUDA_SRC = os.path.join(_ROOT, "cuda")

_SOURCES = [
    os.path.join(_CUDA_SRC, "hash", "hash.cu"),
    os.path.join(_CUDA_SRC, "spconv", "morton.cu"),
    os.path.join(_CUDA_SRC, "spconv", "neighbor_map.cu"),
    os.path.join(_CUDA_SRC, "ext.cpp"),
]

_JIT_NAME = "flex_gemm_kernels_cuda"
_module: Any | None = None
_load_lock = threading.Lock()


def _build_compile_flags() -> tuple[list[str], list[str], bool]:
    build_target = os.environ.get("BUILD_TARGET", "auto")
    if build_target == "auto":
        is_hip = bool(IS_HIP_EXTENSION)
    else:
        is_hip = build_target == "rocm"

    if not is_hip:
        cc_flag = ["--use_fast_math"]
    else:
        archs = os.getenv("GPU_ARCHS", "native").split(";")
        cc_flag = [f"--offload-arch={arch}" for arch in archs]

    if platform.system() == "Windows":
        extra_cxx = [
            "/O2",
            "/std:c++17",
            "/EHsc",
            "/openmp",
            "/permissive-",
            "/Zc:__cplusplus",
        ]
        extra_cuda = (
            [
                "-O3",
                "-std=c++17",
                "-Xcompiler=/std:c++17",
                "-Xcompiler=/EHsc",
                "-Xcompiler=/permissive-",
                "-Xcompiler=/Zc:__cplusplus",
            ]
            + cc_flag
        )
    else:
        cxx11_abi = "1" if torch.compiled_with_cxx11_abi() else "0"
        extra_cxx = [
            "-O3",
            "-std=c++17",
            "-fopenmp",
            f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
        ]
        extra_cuda = ["-O3", "-std=c++17"] + cc_flag

    return extra_cxx, extra_cuda, is_hip


def load_cuda_extension() -> Any:
    """Load (JIT-compile if needed) the pybind CUDA extension module."""
    global _module
    if _module is not None:
        return _module

    if os.environ.get("FLEX_GEMM_DISABLE_CUDA_JIT", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        msg = (
            "FLEX_GEMM_DISABLE_CUDA_JIT is set; refusing to JIT-compile the FlexGEMM "
            "CUDA extension."
        )
        raise RuntimeError(msg)

    with _load_lock:
        if _module is not None:
            return _module

        extra_cxx, extra_cuda, _ = _build_compile_flags()
        verbose = os.environ.get("FLEX_GEMM_VERBOSE_JIT", "").lower() in (
            "1",
            "true",
            "yes",
        )

        # ROCm builds still consume .cu sources via PyTorch's extension machinery.
        _module = load(
            name=_JIT_NAME,
            sources=_SOURCES,
            extra_include_paths=[_CUDA_SRC],
            extra_cflags=extra_cxx,
            extra_cuda_cflags=extra_cuda,
            verbose=verbose,
            with_cuda=True,
        )
        return _module
