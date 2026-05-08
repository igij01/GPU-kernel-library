"""qkv_bmm_split problem — registers problem and kernels on import."""

from . import problem  # noqa: F401 — registers QkvBmmSplitProblem
from .kernels import triton  # noqa: F401 — registers Triton kernel
