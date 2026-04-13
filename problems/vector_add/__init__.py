"""vector_add problem — registers problem and all kernels on import."""

from . import problem  # noqa: F401 — registers VectorAddProblem
from .kernels import cuda, triton  # noqa: F401 — registers both kernels
