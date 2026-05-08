"""Triton qkv_bmm_split kernel — two-pass baseline.

Computes (per program) BLOCK_BG x K output of one of q/k/v at one m position.

Reference (from problem.py):
    qkv = (A.transpose(0,1) @ B).transpose(0,1)         # (Bg, M, 3K)
    q, k, v = split(qkv, K, dim=-1)
    q = rmsnorm(q, norm_1_weight); k = rmsnorm(k, norm_2_weight); v = v
    out = reshape(Bg, M, Nh, K/Nh).transpose(1, 2)      # (Bg, Nh, M, K/Nh)

Grid: (cdiv(Bg, BLOCK_BG), M, 3)   — last axis selects q (0), k (1), v (2).

Loop:
  for k1 in range(0, K, BLOCK_K1):
      acc = 0
      for k2 in range(0, K, BLOCK_K2):
          a  = A[bg_block, m, k2:k2+BLOCK_K2]              # (BLOCK_BG, BLOCK_K2)
          b  = B[m, k2:k2+BLOCK_K2, qkv*K + k1:k1+BLOCK_K1] # (BLOCK_K2, BLOCK_K1)
          acc += a @ b
      if norm: norm_acc += sum(acc*acc, axis=1)
      store acc to output (un-normalised)
  if norm:
      rms = sqrt(norm_acc / K + eps)
      for k1: reload, divide by rms[:, None], multiply by weight, store
"""

import math

import triton
import triton.language as tl

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_TARGET_ARCHS = [CUDAArch.COMPUTE_80]


def _grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    Bg = sizes["Bg"]
    M = sizes["M"]
    BLOCK_BG = config.params["BLOCK_BG"]
    return GridResult(grid=(math.ceil(Bg / BLOCK_BG), M, 3))


@Registry.kernel(
    "qkv_bmm_split_triton",
    backend="triton",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={
        "config_space": {
            "BLOCK_BG": [16, 32, 64],
            "BLOCK_K1": [32, 64, 128],
            "BLOCK_K2": [32, 64],
            "num_stages": [2, 3, 4],
            "num_warps": [4, 8],
        },
        "num_outputs": 3,
    },
    problem="qkv_bmm_split",
    runtime_args=["Bg", "M", "K", "Nh"],
)
@triton.jit
def qkv_bmm_split_kernel(
    A_ptr,            # (Bg, M, K)
    B_ptr,            # (M, K, 3*K)
    NW1_ptr,          # (K,) — norm weight for q
    NW2_ptr,          # (K,) — norm weight for k
    Q_ptr,            # (Bg, Nh, M, K/Nh)
    K_ptr,            # (Bg, Nh, M, K/Nh)
    V_ptr,            # (Bg, Nh, M, K/Nh)
    Bg, M, K, Nh,
    BLOCK_BG: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    pid_bg = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_qkv = tl.program_id(2)            # 0 = q, 1 = k, 2 = v

    EPS: tl.constexpr = 1e-6

    bg_offs = pid_bg * BLOCK_BG + tl.arange(0, BLOCK_BG)
    bg_mask = bg_offs < Bg

    Kh = K // Nh
    qkv_col_off = pid_qkv * K

    is_q = pid_qkv == 0
    is_k = pid_qkv == 1
    is_v = pid_qkv == 2
    do_norm = is_q or is_k

    bg_stride_out = K * M

    # ---- Pass 1: bmm + store un-normalised, accumulate sum-of-squares ----
    norm_acc = tl.zeros([BLOCK_BG], dtype=tl.float32)

    for k1_start in range(0, K, BLOCK_K1):
        k1_offs = k1_start + tl.arange(0, BLOCK_K1)
        k1_mask = k1_offs < K

        acc = tl.zeros([BLOCK_BG, BLOCK_K1], dtype=tl.float32)

        for k2_start in range(0, K, BLOCK_K2):
            k2_offs = k2_start + tl.arange(0, BLOCK_K2)
            k2_mask = k2_offs < K

            a_ptrs = A_ptr + bg_offs[:, None] * (M * K) + pid_m * K + k2_offs[None, :]
            a = tl.load(a_ptrs, mask=bg_mask[:, None] & k2_mask[None, :], other=0.0)

            b_ptrs = (
                B_ptr
                + pid_m * (K * 3 * K)
                + k2_offs[:, None] * (3 * K)
                + (qkv_col_off + k1_offs)[None, :]
            )
            b = tl.load(b_ptrs, mask=k2_mask[:, None] & k1_mask[None, :], other=0.0)

            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        # Match reference rmsnorm numerics by round-tripping through bf16.
        acc_q = acc.to(Q_ptr.dtype.element_ty).to(tl.float32)
        norm_acc += tl.sum(acc_q * acc_q, axis=1)

        nh = k1_offs // Kh
        kh = k1_offs % Kh
        out_col_addr = nh * (M * Kh) + pid_m * Kh + kh
        out_ptrs_base = bg_offs[:, None] * bg_stride_out + out_col_addr[None, :]
        store_mask = bg_mask[:, None] & k1_mask[None, :]

        if is_q:
            tl.store(Q_ptr + out_ptrs_base, acc.to(Q_ptr.dtype.element_ty), mask=store_mask)
        if is_k:
            tl.store(K_ptr + out_ptrs_base, acc.to(K_ptr.dtype.element_ty), mask=store_mask)
        if is_v:
            tl.store(V_ptr + out_ptrs_base, acc.to(V_ptr.dtype.element_ty), mask=store_mask)

    # ---- Pass 2: normalise (q and k only) ----
    if do_norm:
        mean_sq = (norm_acc / K.to(tl.float32)).to(Q_ptr.dtype.element_ty).to(tl.float32)
        rms = tl.sqrt(mean_sq + EPS)

        for k1_start in range(0, K, BLOCK_K1):
            k1_offs = k1_start + tl.arange(0, BLOCK_K1)
            k1_mask = k1_offs < K

            nh = k1_offs // Kh
            kh = k1_offs % Kh
            out_col_addr = nh * (M * Kh) + pid_m * Kh + kh
            out_ptrs_base = bg_offs[:, None] * bg_stride_out + out_col_addr[None, :]
            mask = bg_mask[:, None] & k1_mask[None, :]

            if is_q:
                w = tl.load(NW1_ptr + k1_offs, mask=k1_mask, other=0.0)
                vals = tl.load(Q_ptr + out_ptrs_base, mask=mask, other=0.0).to(tl.float32)
                normalised = vals / rms[:, None] * w[None, :].to(tl.float32)
                tl.store(Q_ptr + out_ptrs_base, normalised.to(Q_ptr.dtype.element_ty), mask=mask)
            if is_k:
                w = tl.load(NW2_ptr + k1_offs, mask=k1_mask, other=0.0)
                vals = tl.load(K_ptr + out_ptrs_base, mask=mask, other=0.0).to(tl.float32)
                normalised = vals / rms[:, None] * w[None, :].to(tl.float32)
                tl.store(K_ptr + out_ptrs_base, normalised.to(K_ptr.dtype.element_ty), mask=mask)
