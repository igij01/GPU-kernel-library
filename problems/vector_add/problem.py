"""VectorAdd problem — element-wise addition of two vectors.

Reference implementation uses PyTorch's built-in addition.
Registering with @Registry.problem at import time.
"""

import torch

import kernel_pipeline_backend.backends.cuda    # noqa: F401 — registers CUDA backend
import kernel_pipeline_backend.backends.triton  # noqa: F401 — registers Triton backend

from kernel_pipeline_backend.problem import rand_tensor
from kernel_pipeline_backend.registry import Registry


@Registry.problem("vector_add")
class VectorAddProblem:
    """C = A + B for 1-D float32 vectors."""

    sizes = {
        "N": [1024, 4096, 16384, 65536, 262144, 1048576],
    }
    dtypes = [torch.float32]
    atol = 1e-5
    rtol = 1e-5

    def initialize(
        self,
        sizes: dict[str, int],
        dtype: torch.dtype | None = None,
    ) -> list[torch.Tensor]:
        dtype = dtype or torch.float32
        N = sizes["N"]
        A = rand_tensor(N, dtype=dtype, device="cuda")
        B = rand_tensor(N, dtype=dtype, device="cuda")
        C = torch.empty(N, dtype=dtype, device="cuda")
        return [A, B, C]

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
    ) -> list[torch.Tensor]:
        A, B, _C = inputs
        return [A + B]
