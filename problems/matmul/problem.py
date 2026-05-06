"""MatMul problem — matrix multiplication C = A @ B.

A: (M, K) InputT, B: (K, N) InputT → C: (M, N) float32.
Reference uses PyTorch's float32 matmul for maximum accuracy.

The ``dtypes`` sweep controls the input precision (InputT).  The output
buffer C is always float32 (the accumulator type).

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
    """C = A @ B  (InputT inputs, fp32 accumulator output)."""

    sizes = {
        "M": [128, 256, 512],
        "N": [128, 256, 512],
        "K": [128, 256],
    }
    # Sweep over input precisions; output is always fp32.
    dtypes = [torch.float16]
    # fp16 → fp32 accumulated matmul; allow a little slack against the
    # pure-fp32 reference.
    atol = 0.05
    rtol = 0.01

    def initialize(
        self,
        sizes: dict[str, int],
        dtype: torch.dtype | None = None,
    ) -> list[torch.Tensor]:
        dtype = dtype or torch.float16
        M, N, K = sizes["M"], sizes["N"], sizes["K"]
        A = rand_tensor(M, K, dtype=dtype, device="cuda")
        B = rand_tensor(K, N, dtype=dtype, device="cuda")
        C = torch.empty((M, N), dtype=torch.float32, device="cuda")
        return [A, B, C]

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
    ) -> list[torch.Tensor]:
        A, B, _C = inputs
        # Up-cast to float32 so the reference isn't subject to rounding.
        return [A.float() @ B.float()]
