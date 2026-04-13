"""self_attention problem — registers problem and all kernels on import."""
from . import problem          # noqa: F401
from .kernels import cuda, triton  # noqa: F401
