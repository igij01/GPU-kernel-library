"""Triton vector addition kernel — registered with the framework on import."""

import math

import triton
import triton.language as tl

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_SIZES = [64, 128, 256, 512, 1024]
_TARGET_ARCHS = [CUDAArch.COMPUTE_80]


def _grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Launch ceil(N / BLOCK_SIZE) programs (Triton manages block dims)."""
    return GridResult(grid=(math.ceil(sizes["N"] / config.params["BLOCK_SIZE"]),))


@Registry.kernel(
    "vector_add_triton",
    backend="triton",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={"config_space": {"BLOCK_SIZE": _BLOCK_SIZES}},
    problem="vector_add",
    runtime_args=["N"],
)
@triton.jit
def vector_add_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)
