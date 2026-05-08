"""debug.py — single-point debugging for qkv_bmm_split kernels.

Usage:
    python problems/qkv_bmm_split/debug.py
    python problems/qkv_bmm_split/debug.py --Bg 1000 --M 16 --K 1024 --Nh 4
    python problems/qkv_bmm_split/debug.py --no-verify
    python problems/qkv_bmm_split/debug.py --no-profile
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import problems.qkv_bmm_split  # noqa: F401, E402

from kernel_pipeline_backend.core.types import KernelConfig, SearchPoint
from kernel_pipeline_backend.device import DeviceHandle
from kernel_pipeline_backend.service import TuneService
from kernel_pipeline_backend.storage import DatabaseStore

_TRITON_KERNEL = "qkv_bmm_split_triton"


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
        print(f"Verification: FAILED — {result.verification.failure}")

    if result.profile_result is None:
        print("Profiling   : skipped")
    else:
        ar = result.profile_result
        print(f"Profiling   : {ar.time_ms:.4f} ms")
        if ar.metrics:
            for k, v in ar.metrics.items():
                print(f"  {k}: {v}")


async def _run(
    kernel_name: str,
    Bg: int, M: int, K: int, Nh: int,
    block_bg: int, block_k1: int, block_k2: int,
    verify: bool, profile: bool,
) -> None:
    service = TuneService(
        device=DeviceHandle(0),
        store=DatabaseStore("sqlite://"),
    )
    point = SearchPoint(
        sizes={"Bg": Bg, "M": M, "K": K, "Nh": Nh},
        config=KernelConfig(params={
            "BLOCK_BG": block_bg,
            "BLOCK_K1": block_k1,
            "BLOCK_K2": block_k2,
            "num_stages": 2,
            "num_warps": 8,
        }),
        dtypes={"T": torch.bfloat16},
    )
    print(f"Running single-point debug for '{kernel_name}'...")
    result = await service.run_point(kernel_name, point, verify=verify, profile=profile)
    _print_point_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug a single qkv_bmm_split point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--Bg", type=int, default=1000)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--Nh", type=int, default=4)
    parser.add_argument("--block-bg", type=int, default=16)
    parser.add_argument("--block-k1", type=int, default=128)
    parser.add_argument("--block-k2", type=int, default=64)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--no-profile", dest="profile", action="store_false")

    args = parser.parse_args()
    asyncio.run(_run(
        _TRITON_KERNEL,
        args.Bg, args.M, args.K, args.Nh,
        args.block_bg, args.block_k1, args.block_k2,
        args.verify, args.profile,
    ))


if __name__ == "__main__":
    main()
