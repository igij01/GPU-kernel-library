"""debug.py — single-point debugging for matmul kernels.

Compiles, verifies, and profiles one specific size + config point without
running the full autotune sweep.

Three kernels are available:
    matmul_triton       — Triton tiled matmul (default)
    matmul_cuda_core    — CUDA shared-memory core (--kernel cuda-core)
    matmul_tensor_core  — CUDA WMMA tensor-core   (--kernel tensor-core)

Usage examples:
    python problems/matmul/debug.py
    python problems/matmul/debug.py --kernel cuda-core
    python problems/matmul/debug.py --kernel tensor-core
    python problems/matmul/debug.py --M 512 --N 512 --K 256
    python problems/matmul/debug.py --block-size 32 --kernel cuda-core
    python problems/matmul/debug.py --block-m 128 --block-n 64 --block-k 32
    python problems/matmul/debug.py --no-verify
    python problems/matmul/debug.py --no-profile
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Anchor: two levels up from this file reaches the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Register the matmul problem and all three kernels.
import problems.matmul  # noqa: F401, E402

from kernel_pipeline_backend.core.types import KernelConfig, SearchPoint  # noqa: E402
from kernel_pipeline_backend.device import DeviceHandle                    # noqa: E402
from kernel_pipeline_backend.service import TuneService                    # noqa: E402
from kernel_pipeline_backend.storage import DatabaseStore                  # noqa: E402

_TRITON_KERNEL      = "matmul_triton"
_CUDA_CORE_KERNEL   = "matmul_cuda_core"
_TENSOR_CORE_KERNEL = "matmul_tensor_core"

_KERNEL_CHOICES = {
    "triton":      _TRITON_KERNEL,
    "cuda-core":   _CUDA_CORE_KERNEL,
    "tensor-core": _TENSOR_CORE_KERNEL,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_result(result) -> None:
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
    sizes: dict[str, int],
    config_params: dict[str, int],
    verify: bool,
    profile: bool,
) -> None:
    service = TuneService(
        device=DeviceHandle(0),
        store=DatabaseStore("sqlite://"),
    )
    point = SearchPoint(
        sizes=sizes,
        config=KernelConfig(params=config_params),
    )
    print(f"Running single-point debug for '{kernel_name}'...")
    result = await service.run_point(kernel_name, point, verify=verify, profile=profile)
    _print_result(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug a single matmul (M, N, K, config) point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--kernel", "-k",
        choices=list(_KERNEL_CHOICES.keys()),
        default="triton",
        help="Which kernel to run (default: triton)",
    )
    parser.add_argument("--M", "-m", type=int, default=256, metavar="M")
    parser.add_argument("--N", "-n", type=int, default=256, metavar="N")
    parser.add_argument("--K",       type=int, default=128, metavar="K")

    # Triton-specific block sizes.
    parser.add_argument("--block-m", type=int, default=64,  metavar="BLOCK_M",
                        help="Triton BLOCK_M (default: 64)")
    parser.add_argument("--block-n", type=int, default=64,  metavar="BLOCK_N",
                        help="Triton BLOCK_N (default: 64)")
    parser.add_argument("--block-k", type=int, default=32,  metavar="BLOCK_K",
                        help="Triton BLOCK_K (default: 32)")

    # CUDA kernel block size.
    parser.add_argument("--block-size", "-b", type=int, default=32, metavar="BLOCK_SIZE",
                        help="CUDA BLOCK_SIZE (default: 32)")

    parser.add_argument("--no-verify",  dest="verify",  action="store_false",
                        help="Skip correctness verification")
    parser.add_argument("--no-profile", dest="profile", action="store_false",
                        help="Skip timing / profiling")

    args = parser.parse_args()
    kernel_name = _KERNEL_CHOICES[args.kernel]

    sizes = {"M": args.M, "N": args.N, "K": args.K}

    if args.kernel == "triton":
        config_params = {
            "BLOCK_M": args.block_m,
            "BLOCK_N": args.block_n,
            "BLOCK_K": args.block_k,
        }
    else:
        config_params = {"BLOCK_SIZE": args.block_size}

    asyncio.run(_run(kernel_name, sizes, config_params, args.verify, args.profile))


if __name__ == "__main__":
    main()
