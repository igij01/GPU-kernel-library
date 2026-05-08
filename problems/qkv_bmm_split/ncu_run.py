"""Standalone launcher for the qkv_bmm_split Triton kernel at the best config.

Used by NCU profiling — bypasses the framework so NCU sees a single
kernel launch (after a warmup launch).

Usage (inside the cuda130-torch container):
    ncu --set full -o qkv_bmm_split_report -f \
        python problems/qkv_bmm_split/ncu_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import problems.qkv_bmm_split  # noqa: F401  — registers kernel
from problems.qkv_bmm_split.kernels.triton.qkv_bmm_split import (  # noqa: E402
    qkv_bmm_split_kernel,
)

# Problem sizes (from problem.sizes)
Bg, M, K, Nh = 1000, 16, 1024, 4

# Best config from tune.py — updated after tuning the new design.
BLOCK_BG = 16
BLOCK_K1 = 128
BLOCK_K2 = 64
NUM_STAGES = 3
NUM_WARPS = 8


def main() -> None:
    dtype = torch.bfloat16
    device = "cuda"

    A = torch.rand((Bg, M, K), dtype=dtype, device=device)
    B = torch.rand((M, K, K * 3), dtype=dtype, device=device)
    nw1 = torch.ones((K,), dtype=dtype, device=device)
    nw2 = torch.ones((K,), dtype=dtype, device=device)

    Kh = K // Nh
    q = torch.empty((Bg, Nh, M, Kh), dtype=dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)

    grid = ((Bg + BLOCK_BG - 1) // BLOCK_BG, M, 3)

    def launch():
        qkv_bmm_split_kernel[grid](
            A, B, nw1, nw2, q, k, v,
            Bg, M, Nh,
            K=K,
            BLOCK_BG=BLOCK_BG,
            BLOCK_K1=BLOCK_K1,
            BLOCK_K2=BLOCK_K2,
            num_stages=NUM_STAGES,
            num_warps=NUM_WARPS,
        )

    # Warmup (Triton JIT-compiles on first call — keep it out of NCU capture
    # by exiting the kernel-name filter only on the second launch... but
    # the simplest approach is just to launch once before NCU's --launch-skip).
    # We do a single warmup, then a single profiled launch.
    launch()
    torch.cuda.synchronize()

    # The launch NCU should profile.
    launch()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
