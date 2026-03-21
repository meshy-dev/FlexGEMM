"""
Shared utilities for FlexGEMM custom-op tensor tests.

Provides small test data generators and helpers for checking
``torch.compile`` compatibility (graph breaks, fullgraph, correctness).
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as dynamo
from torch import Tensor


# ---------------------------------------------------------------------------
# Test-size constants (small -- we care about correctness, not perf)
# ---------------------------------------------------------------------------

RES = 32
CH = 64
N_QUERY = 256

# ---------------------------------------------------------------------------
# Colour helpers (ANSI)
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

PASS = f"{_GREEN}PASS{_RESET}"
FAIL = f"{_RED}FAIL{_RESET}"

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


@torch.no_grad()
def sphere_coords(
    res: int = RES,
    ch: int = CH,
    device: str = "cuda",
    dtype: torch.dtype = torch.float,
) -> tuple[Tensor, Tensor, torch.Size]:
    """Generate a sphere-shell sparse tensor for testing.

    Returns:
        feats: [N, ch] features
        coords: [N, 4] int32 (batch, x, y, z)
        shape: torch.Size([1, ch, res, res, res])
    """
    l_coords: list[Tensor] = []
    for i in range(0, res, 256):
        for j in range(0, res, 256):
            for k in range(0, res, 256):
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(i, min(i + 256, res), device=device),
                        torch.arange(j, min(j + 256, res), device=device),
                        torch.arange(k, min(k + 256, res), device=device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).int()
                dist = ((grid.float() - res / 2 + 0.5) ** 2).sum(-1).sqrt()
                active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
                xyz = torch.nonzero(active).int() + torch.tensor(
                    [i, j, k], device=device, dtype=torch.int32
                )
                l_coords.append(xyz)
    coords = torch.cat(l_coords, dim=0)
    batch = torch.zeros(
        coords.shape[0], 1, device=device, dtype=torch.int32
    )
    coords = torch.cat([batch, coords], dim=-1).contiguous()
    feats = torch.randn(coords.shape[0], ch, device=device, dtype=dtype)
    shape = torch.Size([1, ch, res, res, res])
    return feats, coords, shape


# ---------------------------------------------------------------------------
# Compile-check helpers
# ---------------------------------------------------------------------------


def check_no_graph_breaks(
    fn: Callable,
    args: tuple,
    name: str,
) -> bool:
    """Run ``dynamo.explain`` and verify 0 graph breaks."""
    dynamo.reset()
    try:
        explanation = dynamo.explain(fn)(*args)
        count = explanation.graph_break_count
        if count == 0:
            print(f"  [explain]    graph breaks = 0  {PASS}")
            return True
        elif count < 0:
            # -1 means explain couldn't determine count (e.g. dynamic sizes).
            # Not a failure -- fullgraph check is the authoritative test.
            print(f"  [explain]    SKIP (indeterminate, count={count})")
            return True
        else:
            print(f"  [explain]    graph breaks = {count}  {FAIL}")
            if hasattr(explanation, "break_reasons"):
                for br in explanation.break_reasons:
                    print(f"               - {str(br)[:140]}")
            return False
    except Exception as exc:
        # dynamo.explain can fail on ops with data-dependent output sizes
        # (new_dynamic_size). Fall through -- fullgraph check is the real test.
        msg = str(exc).split("\n")[0][:120]
        print(f"  [explain]    SKIP (explain raised: {msg})")
        return True


def check_fullgraph(
    fn: Callable,
    args: tuple,
    name: str,
) -> bool:
    """Compile with ``fullgraph=True`` and execute."""
    dynamo.reset()
    try:
        compiled = torch.compile(fn, fullgraph=True)
        compiled(*args)
        torch.cuda.synchronize()
        print(f"  [fullgraph]  {PASS}")
        return True
    except Exception as exc:
        msg = str(exc).split("\n")[0][:200]
        print(f"  [fullgraph]  {FAIL}")
        print(f"               {msg}")
        return False


def check_compiled_correctness(
    fn: Callable,
    args: tuple,
    name: str,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Compare eager vs default-compiled outputs (tensor-by-tensor)."""
    dynamo.reset()
    eager_out = fn(*args)
    torch.cuda.synchronize()

    dynamo.reset()
    compiled_out = torch.compile(fn)(*args)
    torch.cuda.synchronize()

    if isinstance(eager_out, Tensor):
        eager_out = (eager_out,)
    if isinstance(compiled_out, Tensor):
        compiled_out = (compiled_out,)

    for i, (e, c) in enumerate(zip(eager_out, compiled_out)):
        if not isinstance(e, Tensor):
            continue
        if e.is_floating_point():
            if not torch.allclose(e, c, rtol=rtol, atol=atol):
                diff = (e.float() - c.float()).abs().max().item()
                print(f"  [correct]    {FAIL}  output[{i}] max diff = {diff:.6e}")
                return False
        elif not torch.equal(e, c):
            print(f"  [correct]    {FAIL}  output[{i}] integer mismatch")
            return False

    print(f"  [correct]    {PASS}")
    return True


def check_backward_correctness(
    fn: Callable,
    args: tuple,
    name: str,
    bwd_input_idx: int = 0,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Compare eager vs compiled backward gradients."""
    # Eager backward
    dynamo.reset()
    inp_e = args[bwd_input_idx].detach().clone().requires_grad_(True)
    args_e = list(args)
    args_e[bwd_input_idx] = inp_e
    out_e = fn(*args_e)
    loss_e = out_e.sum() if isinstance(out_e, Tensor) else out_e[0].sum()
    loss_e.backward()
    grad_e = inp_e.grad.clone()

    # Compiled backward
    dynamo.reset()
    inp_c = args[bwd_input_idx].detach().clone().requires_grad_(True)
    args_c = list(args)
    args_c[bwd_input_idx] = inp_c
    out_c = torch.compile(fn)(*args_c)
    loss_c = out_c.sum() if isinstance(out_c, Tensor) else out_c[0].sum()
    loss_c.backward()
    grad_c = inp_c.grad.clone()

    if torch.allclose(grad_e, grad_c, rtol=rtol, atol=atol):
        print(f"  [bwd grad]   {PASS}")
        return True
    else:
        diff = (grad_e.float() - grad_c.float()).abs().max().item()
        print(f"  [bwd grad]   {FAIL}  max diff = {diff:.6e}")
        return False


def run_test(
    name: str,
    fn: Callable,
    args: tuple,
    *,
    test_compile: bool = True,
    test_bwd: bool = False,
    bwd_input_idx: int = 0,
) -> bool:
    """Run a full test battery on *fn* and return True if all checks pass."""
    sep = "-" * 64
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)

    ok = True

    if test_compile:
        ok &= check_no_graph_breaks(fn, args, name)
        ok &= check_fullgraph(fn, args, name)
        ok &= check_compiled_correctness(fn, args, name)

    if test_bwd:
        try:
            ok &= check_backward_correctness(
                fn, args, name, bwd_input_idx=bwd_input_idx
            )
        except Exception as exc:
            print(f"  [bwd grad]   {FAIL}  {exc}")
            ok = False

    dynamo.reset()
    return ok


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(results: dict[str, bool]) -> int:
    """Print summary table and return 0 if all passed, 1 otherwise."""
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)

    n_pass = sum(results.values())
    n_total = len(results)

    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  {name:<50} {tag}")

    print("-" * 64)
    print(f"  {n_pass}/{n_total} passed")
    print("=" * 64)

    return 0 if n_pass == n_total else 1
