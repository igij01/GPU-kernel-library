"""GPU kernel problems — importing this package registers all problems and kernels."""

from . import vector_add      # noqa: F401 — triggers problem + kernel registration
from . import self_attention  # noqa: F401
from . import matmul          # noqa: F401
