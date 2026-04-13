"""debug.py — single-point debugging for vector_add kernels.

Compiles, verifies, and profiles one specific (N, BLOCK_SIZE) pair without
running the full autotune sweep.  Useful for checking correctness or timing
at a particular operating point.

Usage:
    python problems/vector_add/debug.py                        # triton, N=65536, BLOCK_SIZE=256
    python problems/vector_add/debug.py --cuda                 # CUDA kernel instead
    python problems/vector_add/debug.py --N 1048576 --block-size 512
    python problems/vector_add/debug.py --no-verify            # skip correctness check
    python problems/vector_add/debug.py --no-profile           # skip timing
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `problems` is importable regardless
# of where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Register vector_add problem and both kernels
import problems.vector_add  # noqa: F401, E402

from kernel_pipeline_backend.core.types import KernelConfig, SearchPoint
from kernel_pipeline_backend.device import DeviceHandle
from kernel_pipeline_backend.service import TuneService
from kernel_pipeline_backend.storage import DatabaseStore

_TRITON_KERNEL = "vector_add_triton"
_CUDA_KERNEL = "vector_add_cuda"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_point_result(result) -> None:
    print(f"\nKernel      : {result.kernel_name}")
    print(f"Sizes       : {result.point.sizes}")
    print(f"Config      : {result.point.config.params}")

    if result.compile_error:
        print(f"Compilation : FAILED — {result.compile_error}")
        return
    print("Compilation : OK")

    if result.verification is None:
        print("Verification: skipped")
    elif result.verification.passed:
        print("Verification: PASSED")
    else:
        print(f"Verification: FAILED — {result.verification.message}")

    if result.profile_result is None:
        print("Profiling   : skipped")
    else:
        ar = result.profile_result
        print(f"Profiling   : {ar.time_ms:.4f} ms")
        if ar.metrics:
            for k, v in ar.metrics.items():
                print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------


async def _run(kernel_name: str, N: int, block_size: int, verify: bool, profile: bool) -> None:
    service = TuneService(
        device=DeviceHandle(0),
        store=DatabaseStore("sqlite://"),
    )
    point = SearchPoint(
        sizes={"N": N},
        config=KernelConfig(params={"BLOCK_SIZE": block_size}),
    )
    print(f"Running single-point debug for '{kernel_name}'...")
    result = await service.run_point(kernel_name, point, verify=verify, profile=profile)
    _print_point_result(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug a single vector_add (N, BLOCK_SIZE) point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help=f"Use the CUDA kernel ({_CUDA_KERNEL}) instead of Triton ({_TRITON_KERNEL})",
    )
    parser.add_argument(
        "--N", "-n",
        type=int,
        default=65536,
        metavar="N",
        help="Vector length (default: 65536)",
    )
    parser.add_argument(
        "--block-size", "-b",
        type=int,
        default=256,
        metavar="BLOCK_SIZE",
        help="BLOCK_SIZE config param (default: 256)",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip correctness verification against the PyTorch reference",
    )
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="Skip timing / profiling",
    )

    args = parser.parse_args()
    kernel = _CUDA_KERNEL if args.cuda else _TRITON_KERNEL
    asyncio.run(_run(kernel, args.N, args.block_size, args.verify, args.profile))


if __name__ == "__main__":
    main()
