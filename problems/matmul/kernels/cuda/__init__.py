"""CUDA matmul kernels — both kernels registered with the framework on import.

matmul_cuda_core    — tiled shared-memory kernel (C++ template on BLOCK_SIZE).
matmul_tensor_core  — WMMA tensor-core kernel   (C++ template on BLOCK_SIZE).

Both kernels share the same grid convention:
    grid  = (ceil(N / BLOCK_SIZE), ceil(M / BLOCK_SIZE))
The block shape differs:
    cuda_core   : (BLOCK_SIZE, BLOCK_SIZE)   — one thread per output element
    tensor_core : ((BLOCK_SIZE/16)² × 32,)  — one warp per 16×16 WMMA tile
"""

import math
from pathlib import Path

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_TARGET_ARCHS = [CUDAArch.SM_120]

_source_core   = (Path(__file__).parent / "matmul_core.cu").read_text()
_source_tensor = (Path(__file__).parent / "matmul_tensor_core.cu").read_text()


# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------

def _grid_core(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """2-D grid of BLOCK_SIZE×BLOCK_SIZE thread blocks."""
    BS = config.params["BLOCK_SIZE"]
    return GridResult(
        grid=(math.ceil(sizes["N"] / BS), math.ceil(sizes["M"] / BS)),
        block=(BS, BS),
    )


def _grid_tensor_core(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """2-D grid; each block holds (BLOCK_SIZE/16)² warps = (BLOCK_SIZE/16)²×32 threads."""
    BS = config.params["BLOCK_SIZE"]
    threads = (BS // 16) ** 2 * 32
    return GridResult(
        grid=(math.ceil(sizes["N"] / BS), math.ceil(sizes["M"] / BS)),
        block=(threads,),
    )


# ---------------------------------------------------------------------------
# Kernel registrations
# ---------------------------------------------------------------------------

# CUDA core: tile sizes 16 and 32.
# (32×32 = 1024 threads/block — the per-block limit.)
# InputT is a type template parameter bound to the problem's dtype sweep.
Registry.register_kernel(
    "matmul_cuda_core",
    source=_source_core,
    backend="cuda",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid_core,
    compile_flags={
        "entry_point": "matmul_core",
        "template_params": ["BLOCK_SIZE", "InputT"],
        "type_params": ["InputT"],
        "config_space": {"BLOCK_SIZE": [16, 32]},
    },
    problem="matmul",
    runtime_args=["M", "N", "K"],
    type_args=["InputT"],
)

# Tensor core: tile sizes 16, 32, 64.
# Threads/block: 32, 128, 512 respectively — all within limits.
# InputT is a type template parameter bound to the problem's dtype sweep.
Registry.register_kernel(
    "matmul_tensor_core",
    source=_source_tensor,
    backend="cuda",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid_tensor_core,
    compile_flags={
        "entry_point": "matmul_tensor_core",
        "template_params": ["BLOCK_SIZE", "InputT"],
        "type_params": ["InputT"],
        "config_space": {"BLOCK_SIZE": [16, 32, 64]},
    },
    problem="matmul",
    runtime_args=["M", "N", "K"],
    type_args=["InputT"],
)
