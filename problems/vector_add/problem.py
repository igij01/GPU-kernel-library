"""VectorAdd problem — element-wise addition of two vectors.

Reference implementation uses PyTorch's built-in addition.
"""

import torch

from kernel_pipeline_backend.problem import rand_tensor


class VectorAddProblem:
    """C = A + B for 1-D float32 vectors."""

    sizes = {
        "N": [1024, 4096, 16384, 65536, 262144, 1048576],
    }
    dtypes = [torch.float32, torch.float32]
    atol = 1e-5
    rtol = 1e-5

    def initialize(self, sizes: dict[str, int]) -> list[torch.Tensor]:
        N = sizes["N"]
        A = rand_tensor(N, dtype=self.dtypes[0], device="cuda")
        B = rand_tensor(N, dtype=self.dtypes[1], device="cuda")
        C = torch.empty(N, dtype=self.dtypes[0], device="cuda")
        return [A, B, C]

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
    ) -> list[torch.Tensor]:
        A, B, _C = inputs
        return [A + B]
