"""matmul problem — registers the problem and all kernels on import."""

from . import problem                # noqa: F401
from .kernels import cuda, triton   # noqa: F401
