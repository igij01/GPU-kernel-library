"""debug.py — single-point debugging for self_attention kernels.

Compiles, verifies, and profiles one specific (S, D, BLOCK_SIZE) triple
without running the full autotune sweep.

Usage:
    python problems/self_attention/debug.py                         # triton, S=128, D=64, BLOCK_SIZE=32
    python problems/self_attention/debug.py --cuda                  # CUDA kernel instead
    python problems/self_attention/debug.py --S 512 --block-size 64
    python problems/self_attention/debug.py --no-verify             # skip correctness check
    python problems/self_attention/debug.py --no-profile            # skip timing
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `problems` is importable regardless
# of where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Register self_attention problem and both kernels
import problems.self_attention  # noqa: F401, E402

from kernel_pipeline_backend.core.types import KernelConfig, SearchPoint
from kernel_pipeline_backend.device import DeviceHandle
from kernel_pipeline_backend.service import TuneService
from kernel_pipeline_backend.storage import DatabaseStore

_TRITON_KERNEL = "self_attention_triton"
_CUDA_KERNEL   = "self_attention_cuda"


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


async def _run(
    kernel_name: str,
    S: int,
    D: int,
    block_size: int,
    verify: bool,
    profile: bool,
) -> None:
    service = TuneService(
        device=DeviceHandle(0),
        store=DatabaseStore("sqlite://"),
    )
    # Triton kernel uses BLOCK_S; CUDA kernel uses BLOCK_SIZE.
    param_key = "BLOCK_SIZE" if kernel_name == _CUDA_KERNEL else "BLOCK_S"
    point = SearchPoint(
        sizes={"S": S, "D": D},
        config=KernelConfig(params={param_key: block_size}),
    )
    print(f"Running single-point debug for '{kernel_name}'...")
    result = await service.run_point(kernel_name, point, verify=verify, profile=profile)
    _print_point_result(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug a single self_attention (S, D, BLOCK_SIZE) point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help=f"Use the CUDA kernel ({_CUDA_KERNEL}) instead of Triton ({_TRITON_KERNEL})",
    )
    parser.add_argument(
        "--S", "-s",
        type=int,
        default=128,
        metavar="S",
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--D", "-d",
        type=int,
        default=64,
        metavar="D",
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--block-size", "-b",
        type=int,
        default=32,
        metavar="BLOCK_SIZE",
        help="BLOCK_SIZE config param (default: 32)",
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
    asyncio.run(_run(kernel, args.S, args.D, args.block_size, args.verify, args.profile))


if __name__ == "__main__":
    main()
