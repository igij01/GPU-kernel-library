"""CUDA vector addition kernel — registered with the framework on import."""

import math
from pathlib import Path

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_SIZES = [64, 128, 256, 512, 1024]
_TARGET_ARCHS = [CUDAArch.COMPUTE_80]

_source = (Path(__file__).parent / "vector_add.cu").read_text()


def _grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Launch ceil(N / BLOCK_SIZE) blocks of BLOCK_SIZE threads."""
    return GridResult(
        grid=(math.ceil(sizes["N"] / config.params["BLOCK_SIZE"]),),
        block=(config.params["BLOCK_SIZE"],),
    )


Registry.register_kernel(
    "vector_add_cuda",
    source=_source,
    backend="cuda",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={
        "entry_point": "vector_add",
        "config_space": {"BLOCK_SIZE": _BLOCK_SIZES},
    },
    problem="vector_add",
    runtime_args=["N"],
)
