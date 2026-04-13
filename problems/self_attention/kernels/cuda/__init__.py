"""CUDA self-attention kernel — registered with the framework on import."""

from pathlib import Path

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_SIZES = [32, 64, 128]
_TARGET_ARCHS = [CUDAArch.SM_120]

_source = (Path(__file__).parent / "self_attention.cu").read_text()


def _grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """One block per query row; threads split the head dimension."""
    return GridResult(
        grid=(sizes["S"],),
        block=(config.params["BLOCK_SIZE"],),
    )


Registry.register_kernel(
    "self_attention_cuda",
    source=_source,
    backend="cuda",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={
        "entry_point": "self_attention",
        "config_space": {"BLOCK_SIZE": _BLOCK_SIZES},
    },
    problem="self_attention",
    runtime_args=["S", "D"],
)
