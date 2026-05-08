"""qkv_bmm_split problem — 
    qkv = transpose(transpose(A) @ B)
    q,k,v = split(qkv, dim=-1)
    q = separate_heads(norm(q))
    k = separate_heads(norm(k))
    v = separate_heads(v)

Registering with @Registry.problem at import time.
"""

import torch

import kernel_pipeline_backend.backends.cuda    # noqa: F401 — registers CUDA backend
import kernel_pipeline_backend.backends.triton  # noqa: F401 — registers Triton backend

from kernel_pipeline_backend.problem import rand_tensor, ones_tensor
from kernel_pipeline_backend.registry import Registry


@Registry.problem("qkv_bmm_split")
class QkvBmmSplitProblem:
    sizes = {
        "Bg": [1000],
        "M": [16],
        "K": [1024],
        "Nh": [4],
    }
    dtypes = [{"T": torch.bfloat16}]
    atol = 1e-2
    rtol = 1e-2

    def initialize(
        self,
        sizes: dict[str, int],
        dtypes: dict[str, torch.dtype],
    ) -> list[torch.Tensor]:
        Bg, M, K, Nh = sizes["Bg"], sizes["M"], sizes["K"], sizes["Nh"]
        dtype = dtypes["T"]
        A = rand_tensor(Bg, M, K, dtype=dtype)
        B = rand_tensor(M, K, K*3, dtype=dtype)
        norm_1_weight = ones_tensor(K, dtype=dtype)
        norm_2_weight = ones_tensor(K, dtype=dtype)
        
        q = torch.empty((Bg, Nh, M, K // Nh), dtype=dtype, device="cuda")
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        
        return [A, B, norm_1_weight, norm_2_weight, q, k, v]
    
    def rmsnorm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float = 1e-6
    ) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)
        output = input / rms
        if weight is not None:
            output = output * weight
        return output

    def reference(
        self,
        inputs: list[torch.Tensor],
        sizes: dict[str, int],
        dtypes: dict[str, torch.dtype],
    ) -> list[torch.Tensor]:
        A, B, norm_1_weight, norm_2_weight, _, _, _ = inputs
        Bg, M, K, Nh = sizes["Bg"], sizes["M"], sizes["K"], sizes["Nh"]
        
        A = A.transpose(0, 1) # M, bg, K
        qkv = A @ B # M, bg, K*3
        qkv = qkv.transpose(0, 1) # bg, M, K*3
        
        query, key, value = torch.split(qkv, split_size_or_sections=K, dim=-1)
        
        query = self.rmsnorm(query, norm_1_weight)
        key = self.rmsnorm(key, norm_2_weight)
        
        separate_heads = lambda x : x.reshape(Bg, M, Nh, K // Nh).transpose(1, 2)
        
        query = separate_heads(query)
        key = separate_heads(key)
        value = separate_heads(value)
        
        return [query, key, value]
