"""Triton qkv_bmm_split kernel — bf16 row staging across the k1 outer loop.

Goal: avoid the L2 round-trip of the two-pass baseline by keeping the
full per-row bmm result on-chip until rmsnorm has been applied, then
storing once.

Strategy:
  - Persistent bf16 staging buffer shaped ``(BLOCK_BG, NUM_K1, BLOCK_K1)``,
    half the footprint of an fp32 buffer of the same logical size.
  - Outer loop over ``k1_idx`` accumulates one ``(BLOCK_BG, BLOCK_K1)``
    fp32 tile via the inner k2 reduction, casts to bf16, then merges
    into the staging buffer via a one-hot mask along the middle axis.
  - After the bmm, reduce the staging buffer to compute rms, fuse
    divide+weight, and store once in the (Bg, Nh, M, Kh) layout.

Constraints: K must be divisible by BLOCK_K1.
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
    "qkv_bmm_split_triton_bf16_staged",
    backend="triton",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={
        "config_space": {
            "BLOCK_BG": [8, 16],
            "BLOCK_K1": [64, 128],
            "BLOCK_K2": [32, 64],
            "num_stages": [2, 3],
            "num_warps": [4, 8],
        },
        "num_outputs": 3,
    },
    problem="qkv_bmm_split",
    runtime_args=["Bg", "M", "Nh"],
    constexpr_args={"K": "K"},
)
@triton.jit
def qkv_bmm_split_bf16_staged_kernel(
    A_ptr,            # (Bg, M, K)             bf16
    B_ptr,            # (M, K, 3*K)            bf16
    NW1_ptr,          # (K,)                   bf16
    NW2_ptr,          # (K,)                   bf16
    Q_ptr,            # (Bg, Nh, M, K/Nh)      bf16
    K_ptr,
    V_ptr,
    Bg, M, Nh,
    K: tl.constexpr,
    BLOCK_BG: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    pid_bg = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_qkv = tl.program_id(2)            # 0=q, 1=k, 2=v

    EPS: tl.constexpr = 1e-6
    NUM_K1: tl.constexpr = K // BLOCK_K1

    bg_offs = pid_bg * BLOCK_BG + tl.arange(0, BLOCK_BG)
    bg_mask = bg_offs < Bg

    Kh = K // Nh
    qkv_col_off = pid_qkv * K
    k_full = tl.arange(0, K)
    blk_idx = tl.arange(0, NUM_K1)
    inner_idx = tl.arange(0, BLOCK_K1)

    is_q = pid_qkv == 0
    is_k = pid_qkv == 1
    is_v = pid_qkv == 2
    do_norm = is_q or is_k

    acc_bf16 = tl.zeros([BLOCK_BG, NUM_K1, BLOCK_K1], dtype=Q_ptr.dtype.element_ty)

    for k1_idx in range(NUM_K1):
        k1_start = k1_idx * BLOCK_K1
        k1_offs = k1_start + inner_idx

        tile = tl.zeros([BLOCK_BG, BLOCK_K1], dtype=tl.float32)

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
            b = tl.load(b_ptrs, mask=k2_mask[:, None], other=0.0)

            tile = tl.dot(a, b, acc=tile, out_dtype=tl.float32)

        slot_mask = (blk_idx[None, :, None] == k1_idx)
        tile_bf16 = tile.to(Q_ptr.dtype.element_ty)
        acc_bf16 = tl.where(slot_mask, tile_bf16[:, None, :], acc_bf16)

    acc_bf16_flat = tl.reshape(acc_bf16, [BLOCK_BG, K])

    if do_norm:
        acc_f32 = acc_bf16_flat.to(tl.float32)
        sum_sq = tl.sum(acc_f32 * acc_f32, axis=1)
        mean_sq = (sum_sq / float(K)).to(Q_ptr.dtype.element_ty).to(tl.float32)
        rms = tl.sqrt(mean_sq + EPS)

        if is_q:
            w = tl.load(NW1_ptr + k_full).to(tl.float32)
        else:
            w = tl.load(NW2_ptr + k_full).to(tl.float32)
        out_val = (acc_f32 / rms[:, None] * w[None, :]).to(Q_ptr.dtype.element_ty)
    else:
        out_val = acc_bf16_flat

    nh = k_full // Kh
    kh = k_full % Kh
    out_col_addr = nh * (M * Kh) + pid_m * Kh + kh
    out_ptrs_base = bg_offs[:, None] * (K * M) + out_col_addr[None, :]
    store_mask = bg_mask[:, None]

    if is_q:
        tl.store(Q_ptr + out_ptrs_base, out_val, mask=store_mask)
    if is_k:
        tl.store(K_ptr + out_ptrs_base, out_val, mask=store_mask)
    if is_v:
        tl.store(V_ptr + out_ptrs_base, out_val, mask=store_mask)
