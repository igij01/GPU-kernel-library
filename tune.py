"""tune.py — CLI entry point for running the kernel tuning pipeline.

Usage:
    python tune.py                  # tune all registered problems
    python tune.py vector_add       # tune one specific problem
    python tune.py --list           # list all registered problems and kernels
    python tune.py --list vector_add  # list kernels for one problem
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# ---------------------------------------------------------------------------
# Import problems package — registers all problems and kernels as a side effect
# ---------------------------------------------------------------------------

import problems  # noqa: F401

from kernel_pipeline_backend.core.types import KernelConfig
from kernel_pipeline_backend.device import DeviceHandle
from kernel_pipeline_backend.registry import Registry
from kernel_pipeline_backend.service import TuneService
from kernel_pipeline_backend.storage import DatabaseStore
from kernel_pipeline_backend.autotuner.strategy import Exhaustive

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_result(result) -> None:
    pr = result.pipeline_result
    print(f"--- {result.kernel_names} (problem: {result.problem_name}) ---")
    print(f"  Verified : {len(pr.verified)} points")
    print(f"  Autotuned: {len(pr.autotuned)} points")
    print(f"  Skipped  : {len(pr.skipped)}")
    print(f"  Errors   : {len(pr.errors)}")
    if pr.errors:
        for err in pr.errors:
            print(f"    [{err.stage}] {err.message}")

    if pr.autotuned:
        print("  Best configs:")
        by_size: dict[tuple, tuple[float, KernelConfig]] = {}
        for ar in pr.autotuned:
            key = tuple(sorted(ar.point.sizes.items()))
            if key not in by_size or ar.time_ms < by_size[key][0]:
                by_size[key] = (ar.time_ms, ar.point.config)
        for key in sorted(by_size):
            time_ms, cfg = by_size[key]
            sizes_str = ", ".join(f"{k}={v}" for k, v in key)
            cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg.params.items()))
            print(f"    {sizes_str}: {time_ms:.4f} ms  {cfg_str}")
    print()


# ---------------------------------------------------------------------------
# Async tuning
# ---------------------------------------------------------------------------


async def _tune(problem_names: list[str] | None) -> None:
    device = DeviceHandle(0)
    store = DatabaseStore("sqlite://")
    service = TuneService(device=device, store=store, strategy=Exhaustive())

    if problem_names:
        print(f"=== Tuning problem(s): {', '.join(problem_names)} ===\n")
        for prob_name in problem_names:
            kernel_names = Registry.kernels_for_problem(prob_name)
            by_backend: dict[str, list[str]] = {}
            for kname in kernel_names:
                spec = Registry.get_kernel(kname)
                by_backend.setdefault(spec.backend, []).append(kname)

            if len(by_backend) == 1:
                result = await service.tune_problem(prob_name, force=True)
                _print_result(result)
            else:
                # Multiple backends — run each kernel independently
                for kname in kernel_names:
                    result = await service.tune(kname, problem=prob_name, force=True)
                    _print_result(result)
    else:
        print("=== Tuning all problems ===\n")
        results = await service.tune_all(force=True)
        for result in results:
            _print_result(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune GPU kernels through the kernel-pipeline-backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "problems",
        nargs="*",
        metavar="PROBLEM",
        help="Problem name(s) to tune. Omit to tune all.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List registered problems and kernels, then exit.",
    )

    args = parser.parse_args()

    if args.list:
        if args.problems:
            for name in args.problems:
                kernels = Registry.kernels_for_problem(name)
                if not kernels:
                    print(f"{name}: (no kernels registered)")
                else:
                    print(f"{name}:")
                    for k in kernels:
                        spec = Registry.get_kernel(k)
                        print(f"  {k}  [{spec.backend}]")
        else:
            print(Registry.dump_tree())
        return

    # Validate requested problem names
    if args.problems:
        known = set(Registry.list_problems())
        bad = [p for p in args.problems if p not in known]
        if bad:
            print(f"error: unknown problem(s): {', '.join(bad)}", file=sys.stderr)
            print(f"available: {', '.join(sorted(known))}", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_tune(args.problems or None))


if __name__ == "__main__":
    main()
