"""MatMul problem — matrix multiplication C = A @ B.

A: (M, K) float16, B: (K, N) float16 → C: (M, N) float32.
Reference uses PyTorch's float32 matmul for maximum accuracy.

All size values are multiples of 16 so that the WMMA tensor-core kernel
(which requires 16×16×16 tiles) never needs boundary padding.
"""

import torch

import kernel_pipeline_backend.backends.cuda    # noqa: F401 — registers CUDA backend
import kernel_pipeline_backend.backends.triton  # noqa: F401 — registers Triton backend

from kernel_pipeline_backend.problem import rand_tensor
from kernel_pipeline_backend.registry import Registry


@Registry.problem("matmul")
class MatMulProblem:
    """C = A @ B  (fp16 inputs, fp32 output)."""

    sizes = {
        "M": [128, 256, 512],
        "N": [128, 256, 512],
        "K": [128, 256],
    }
    # A and B are float16; the output buffer C is float32.
    dtypes = [torch.float16, torch.float16, torch.float32]
    # fp16 → fp32 accumulated matmul; allow a little slack against the
    # pure-fp32 reference.
    atol = 0.05
    rtol = 0.01

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        M, N, K = sizes["M"], sizes["N"], sizes["K"]
        A = rand_tensor(M, K, dtype=torch.float16, device="cuda")
        B = rand_tensor(K, N, dtype=torch.float16, device="cuda")
        C = torch.empty((M, N), dtype=torch.float32, device="cuda")
        return [A, B, C]

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
    ) -> list[torch.Tensor]:
        A, B, _C = inputs
        # Up-cast to float32 so the reference isn't subject to fp16 rounding.
        return [A.float() @ B.float()]
