"""Tune vectorAdd kernels (CUDA + Triton) through the kernel-pipeline-backend."""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Import and register backends
# ---------------------------------------------------------------------------

import kernel_pipeline_backend.backends.cuda   # noqa: F401 — registers CUDA backend
import kernel_pipeline_backend.backends.triton  # noqa: F401 — registers Triton backend

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.device import DeviceHandle
from kernel_pipeline_backend.registry import Registry
from kernel_pipeline_backend.service import TuneService
from kernel_pipeline_backend.storage import DatabaseStore
from kernel_pipeline_backend.autotuner.strategy import Exhaustive

# ---------------------------------------------------------------------------
# 2. Problem registration
# ---------------------------------------------------------------------------

# Import the problem class from our problems directory
sys.path.insert(0, str(Path(__file__).resolve().parent / "problems" / "vector_add"))
from problem import VectorAddProblem  # noqa: E402

Registry.register_problem("vector_add", VectorAddProblem())

# ---------------------------------------------------------------------------
# 3. Tunable parameter space
# ---------------------------------------------------------------------------

BLOCK_SIZES = [64, 128, 256, 512, 1024]

# ---------------------------------------------------------------------------
# 4. Grid generators
# ---------------------------------------------------------------------------


def cuda_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Launch ceil(N / BLOCK_SIZE) blocks of BLOCK_SIZE threads."""
    N = sizes["N"]
    bs = config.params["BLOCK_SIZE"]
    num_blocks = math.ceil(N / bs)
    return GridResult(grid=(num_blocks,), block=(bs,))


def triton_grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """Launch ceil(N / BLOCK_SIZE) programs (Triton manages block dims)."""
    N = sizes["N"]
    bs = config.params["BLOCK_SIZE"]
    num_blocks = math.ceil(N / bs)
    return GridResult(grid=(num_blocks,))

# ---------------------------------------------------------------------------
# 5. Kernel registration — CUDA
# ---------------------------------------------------------------------------

cuda_source = (
    Path(__file__).resolve().parent
    / "problems" / "vector_add" / "kernels" / "cuda" / "vector_add.cu"
).read_text()

Registry.register_kernel(
    name="vector_add_cuda",
    source=cuda_source,
    backend="cuda",
    target_archs=[CUDAArch.SM_80],
    grid_generator=cuda_grid,
    compile_flags={
        "entry_point": "vector_add",
        "config_space": {"BLOCK_SIZE": BLOCK_SIZES},
    },
    problem="vector_add",
    runtime_args=["N"],
)

# ---------------------------------------------------------------------------
# 6. Kernel registration — Triton
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
        / "problems" / "vector_add" / "kernels" / "triton"
    ),
)
from vector_add import vector_add_kernel  # noqa: E402

Registry.register_kernel(
    name="vector_add_triton",
    source=vector_add_kernel,
    backend="triton",
    target_archs=[CUDAArch.SM_80],
    grid_generator=triton_grid,
    compile_flags={
        "config_space": {"BLOCK_SIZE": BLOCK_SIZES},
    },
    problem="vector_add",
    runtime_args=["N"],
)

# ---------------------------------------------------------------------------
# 7. Run
# ---------------------------------------------------------------------------


async def main() -> None:
    device = DeviceHandle(0)
    store = DatabaseStore("sqlite://")
    service = TuneService(
        device=device,
        store=store,
        strategy=Exhaustive(),
    )

    print("=== Tuning all vectorAdd kernels ===\n")
    results = await service.tune_all(force=True)

    for result in results:
        pr = result.pipeline_result
        print(f"--- {result.kernel_names} (problem: {result.problem_name}) ---")
        print(f"  Verified : {len(pr.verified)} points")
        print(f"  Autotuned: {len(pr.autotuned)} points")
        print(f"  Skipped  : {len(pr.skipped)}")
        print(f"  Errors   : {len(pr.errors)}")
        if pr.errors:
            for err in pr.errors:
                print(f"    [{err.stage}] {err.message}")

        # Show best config per problem size
        if pr.autotuned:
            print("  Best configs:")
            by_size: dict[int, tuple[float, KernelConfig]] = {}
            for ar in pr.autotuned:
                n = ar.point.sizes["N"]
                if n not in by_size or ar.time_ms < by_size[n][0]:
                    by_size[n] = (ar.time_ms, ar.point.config)
            for n in sorted(by_size):
                time_ms, cfg = by_size[n]
                print(f"    N={n:>8d}: {time_ms:.4f} ms  BLOCK_SIZE={cfg.params['BLOCK_SIZE']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
