"""Triton single-head self-attention kernel — registered with the framework on import.

Algorithm: online (single-pass) softmax, one Triton program per query row.

The head dimension D is fixed at 64 (_BLOCK_D constant).  The sequence
dimension S is a runtime value swept via the problem size space.
BLOCK_S controls the tile size for iterating over the key/value sequence.
"""

import math

import triton
import triton.language as tl

from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_SIZES = [16, 32, 64, 128]
_TARGET_ARCHS = [CUDAArch.COMPUTE_80]

# Fixed head dimension — must match D in problem sizes.
# Must be a tl.constexpr so it is accessible inside @triton.jit kernels.
_BLOCK_D = tl.constexpr(64)


def _grid(sizes: dict[str, int], config: KernelConfig) -> GridResult:
    """One Triton program per query row."""
    return GridResult(grid=(sizes["S"],))


@Registry.kernel(
    "self_attention_triton",
    backend="triton",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={"config_space": {"BLOCK_S": _BLOCK_SIZES}},
    problem="self_attention",
    runtime_args=["S", "D"],
)
@triton.jit
def self_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    S,
    D,
    BLOCK_S: tl.constexpr,
):
    """Compute one row of single-head attention output (online softmax)."""
    i = tl.program_id(0)  # query row index

    d_range = tl.arange(0, _BLOCK_D)           # (_BLOCK_D,)
    s_range = tl.arange(0, BLOCK_S)            # (BLOCK_S,)
    scale = 1.0 / tl.sqrt(D.to(tl.float32))

    # Load Q[i, :] — shape (_BLOCK_D,)
    q = tl.load(Q_ptr + i * D + d_range, mask=d_range < D, other=0.0)

    # Running online-softmax statistics and output accumulator
    m_i = float("-inf")                         # running max score
    l_i = 0.0                                   # running denominator
    acc = tl.zeros([_BLOCK_D], dtype=tl.float32)

    # Single pass over K/V tiles
    for j in range(0, tl.cdiv(S, BLOCK_S)):
        j_start = j * BLOCK_S
        s_mask = (j_start + s_range) < S       # (BLOCK_S,)

        # Load K tile: (BLOCK_S, _BLOCK_D)
        k_ptrs = K_ptr + (j_start + s_range[:, None]) * D + d_range[None, :]
        k_block = tl.load(k_ptrs, mask=s_mask[:, None], other=0.0)

        # scores[s] = dot(q, K[j_start+s, :]) * scale — (BLOCK_S,)
        # q[None, :] broadcasts to (BLOCK_S, _BLOCK_D); sum over head dim → (BLOCK_S,)
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale

        # Mask padding positions to -inf so they don't affect softmax
        scores = tl.where(s_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        exp_scores = tl.exp(scores - m_new)     # (BLOCK_S,)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(exp_scores, axis=0)
        acc = acc * tl.exp(m_i - m_new)
        m_i = m_new

        # Load V tile: (BLOCK_S, _BLOCK_D)
        v_ptrs = V_ptr + (j_start + s_range[:, None]) * D + d_range[None, :]
        v_block = tl.load(v_ptrs, mask=s_mask[:, None], other=0.0)

        # acc += exp_scores @ V_tile — (BLOCK_S,) x (BLOCK_S, _BLOCK_D) → (_BLOCK_D,)
        acc = acc + tl.sum(exp_scores[:, None] * v_block, axis=0)

    # Normalise and store O[i, :]
    acc = acc / l_i
    tl.store(O_ptr + i * D + d_range, acc, mask=d_range < D)
