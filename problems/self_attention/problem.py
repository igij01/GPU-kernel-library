"""SelfAttention problem — single-head scaled dot-product attention.

Given Q, K, V of shape (S, D), computes:
    scores = Q @ K^T / sqrt(D)
    attn   = softmax(scores, dim=-1)
    O      = attn @ V

Reference uses torch.nn.functional.scaled_dot_product_attention.
D is fixed at 64 (a standard Transformer head dimension).
"""

import torch
import torch.nn.functional as F

import kernel_pipeline_backend.backends.cuda    # noqa: F401 — registers CUDA backend
import kernel_pipeline_backend.backends.triton  # noqa: F401 — registers Triton backend

from kernel_pipeline_backend.problem import rand_tensor
from kernel_pipeline_backend.registry import Registry


@Registry.problem("self_attention")
class SelfAttentionProblem:
    """Single-head self-attention: O = softmax(QK^T / sqrt(D)) V."""

    sizes = {
        "S": [64, 128, 256, 512],
        "D": [64],
    }
    dtypes = [torch.float32]
    # Softmax + floating-point reductions accumulate more error than simple add
    atol = 1e-3
    rtol = 1e-3

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        S, D = sizes["S"], sizes["D"]
        Q = rand_tensor(S, D, dtype=torch.float32, device="cuda")
        K = rand_tensor(S, D, dtype=torch.float32, device="cuda")
        V = rand_tensor(S, D, dtype=torch.float32, device="cuda")
        O = torch.empty((S, D), dtype=torch.float32, device="cuda")
        return [Q, K, V, O]

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
    ) -> list[torch.Tensor]:
        Q, K, V, _O = inputs
        # sdpa expects (batch, heads, seq, dim); use batch=heads=1
        out = F.scaled_dot_product_attention(
            Q.unsqueeze(0).unsqueeze(0),
            K.unsqueeze(0).unsqueeze(0),
            V.unsqueeze(0).unsqueeze(0),
        )
        return [out.squeeze(0).squeeze(0)]
