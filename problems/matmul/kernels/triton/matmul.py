"""Triton tiled matrix multiplication kernel — registered with the framework on import.

C = A @ B   A:(M,K) fp16, B:(K,N) fp16 → C:(M,N) fp32

Algorithm:
  - One Triton program per (BLOCK_M × BLOCK_N) output tile.
  - Accumulate in fp32 using tl.dot, which uses tensor cores on sm_80+.
  - Grid is 2-D: (ceil(M/BLOCK_M), ceil(N/BLOCK_N)).

Config space: {BLOCK_M} × {BLOCK_N} × {BLOCK_K} — 18 combinations.
tl.dot requires the inner dimension (BLOCK_K) to be ≥ 16 and a power of 2.
"""

import math

import triton
import triton.language as tl

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_MN = [32, 64, 128]
_BLOCK_K  = [32, 64]
_TARGET_ARCHS = [CUDAArch.COMPUTE_80]


def _grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """2-D grid: one program per (BLOCK_M × BLOCK_N) output tile."""
    return GridResult(grid=(
        math.ceil(sizes["M"] / config.params["BLOCK_M"]),
        math.ceil(sizes["N"] / config.params["BLOCK_N"]),
    ))


@Registry.kernel(
    "matmul_triton",
    backend="triton",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={
        "config_space": {
            "BLOCK_M": _BLOCK_MN,
            "BLOCK_N": _BLOCK_MN,
            "BLOCK_K": _BLOCK_K,
        }
    },
    problem="matmul",
    runtime_args=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul: one program computes a (BLOCK_M, BLOCK_N) output patch."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row / column offsets for this tile.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # fp32 accumulator, shape (BLOCK_M, BLOCK_N).
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Sweep over K tiles.
    for k_idx in range(tl.cdiv(K, BLOCK_K)):
        k_offs = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)  # (BLOCK_K,)

        # Load A tile: (BLOCK_M, BLOCK_K) — mask out-of-range entries.
        a = tl.load(
            A_ptr + offs_m[:, None] * K + k_offs[None, :],
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        # Load B tile: (BLOCK_K, BLOCK_N).
        b = tl.load(
            B_ptr + k_offs[:, None] * N + offs_n[None, :],
            mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # tl.dot: fp16 inputs, fp32 accumulation (uses tensor cores on sm_80+).
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    # Write the output tile to C.
    tl.store(
        C_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
