---
name: add-problem
description: Add a new GPU kernel problem to this repo — creates the problem package, Triton and CUDA sample kernels, and a debug script, then runs tune.py and debug.py inside a Docker container to verify everything works end-to-end. Use when the user says "add a problem", "add a kernel", "new problem", or gives a GPU kernel to implement.
---

# add-problem

Adds a new problem + sample kernels to the GPU kernel library, following the same package conventions as `vector_add`.

## Workflow overview

1. Create the problem package directory structure
2. Write `problem.py` with `@Registry.problem` decorator
3. Write the Triton kernel with `@Registry.kernel` decorator
4. Write the CUDA kernel registration in `kernels/cuda/__init__.py`
5. Write a problem-specific `debug.py`
6. Wire the new problem into `problems/__init__.py`
7. Run `tune.py <problem>` in container to verify full pipeline
8. Run `debug.py` in container to verify single-point debug

---

## Step 1 — Gather requirements

Before writing any code, confirm with the user:

- **Problem name** (snake_case, e.g. `matmul`, `softmax`) — used as the Registry key and directory name
- **Size axes** — what dimensions does this problem vary over? (e.g. `M`, `N`, `K` for matmul)
- **Size values** — example sweep values for each axis (e.g. `[128, 256, 512, 1024]`)
- **Dtypes** — input/output tensor dtypes (e.g. `torch.float32`)
- **Reference formula** — what does the correct output look like in PyTorch?
- **CUDA kernel entry point name** — the C function name (e.g. `matmul_naive`)
- **Target arch** — default to `CUDAArch.SM_120` (RTX 5090 in this repo). Adjust if specified.

---

## Step 2 — Create directory structure

```
problems/<problem_name>/
  __init__.py
  problem.py
  debug.py
  kernels/
    __init__.py
    cuda/
      __init__.py
      <problem_name>.cu
    triton/
      __init__.py
      <problem_name>.py
```

All `__init__.py` files under `kernels/` are functional — they trigger registration on import.

---

## Step 3 — Write `problem.py`

Use `@Registry.problem("<problem_name>")` directly on the class. The decorator instantiates the class and registers it — no separate `register_problem()` call needed.

Also import the backends here so they self-register before any kernel registration fires:

```python
import kernel_pipeline_backend.backends.cuda    # noqa: F401
import kernel_pipeline_backend.backends.triton  # noqa: F401

from kernel_pipeline_backend.problem import rand_tensor
from kernel_pipeline_backend.registry import Registry

@Registry.problem("<problem_name>")
class <ProblemClass>:
    sizes = {"DIM": [128, 256, 512, 1024]}
    dtypes = [torch.float32, torch.float32]
    atol = 1e-5
    rtol = 1e-5

    def initialize(self, sizes):
        ...  # return list of tensors (inputs + output buffers)

    def reference(self, inputs, sizes):
        ...  # return list of output tensors from PyTorch ops
```

---

## Step 4 — Write the Triton kernel (`kernels/triton/<problem_name>.py`)

Stack `@Registry.kernel(...)` **above** `@triton.jit`. The decorator receives the `@triton.jit`-wrapped function as `source`, which the hasher and compiler both handle correctly via the `.fn` attribute.

```python
import math
import triton
import triton.language as tl
from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_SIZES = [64, 128, 256, 512, 1024]
_TARGET_ARCHS = [CUDAArch.SM_120]

def _grid(sizes, config):
    return GridResult(grid=(math.ceil(sizes["N"] / config.params["BLOCK_SIZE"]),))

@Registry.kernel(
    "<problem_name>_triton",
    backend="triton",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={"config_space": {"BLOCK_SIZE": _BLOCK_SIZES}},
    problem="<problem_name>",
    runtime_args=["N"],   # size keys forwarded as scalar extra_args
)
@triton.jit
def <problem_name>_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

Then create `kernels/triton/__init__.py`:
```python
from . import <problem_name>  # noqa: F401
```

---

## Step 5 — Write the CUDA kernel

Create `kernels/cuda/<problem_name>.cu` with a standard CUDA C kernel.

Then create `kernels/cuda/__init__.py` — this file reads the `.cu` source and calls `Registry.register_kernel()` imperatively (no decorator possible for string sources):

```python
import math
from pathlib import Path
from kernel_pipeline_backend.core.types import CUDAArch, GridResult, KernelConfig
from kernel_pipeline_backend.registry import Registry

_BLOCK_SIZES = [64, 128, 256, 512, 1024]
_TARGET_ARCHS = [CUDAArch.SM_120]
_source = (Path(__file__).parent / "<problem_name>.cu").read_text()

def _grid(sizes, config):
    return GridResult(
        grid=(math.ceil(sizes["N"] / config.params["BLOCK_SIZE"]),),
        block=(config.params["BLOCK_SIZE"],),
    )

Registry.register_kernel(
    "<problem_name>_cuda",
    source=_source,
    backend="cuda",
    target_archs=_TARGET_ARCHS,
    grid_generator=_grid,
    compile_flags={
        "entry_point": "<cuda_entry_point>",
        "config_space": {"BLOCK_SIZE": _BLOCK_SIZES},
    },
    problem="<problem_name>",
    runtime_args=["N"],
)
```

---

## Step 6 — Write `__init__.py` files

`problems/<problem_name>/__init__.py`:
```python
"""<problem_name> problem — registers problem and all kernels on import."""
from . import problem          # noqa: F401
from .kernels import cuda, triton  # noqa: F401
```

`problems/<problem_name>/kernels/__init__.py`: empty file.

---

## Step 7 — Write `debug.py`

The debug script lives inside the problem directory and is specific to it — no `--kernel` chooser for unrelated kernels. Use `--cuda` as a simple toggle between the problem's two kernels.

Always add a `sys.path` anchor so the script works when invoked from any directory:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import problems.<problem_name>  # noqa: F401, E402
```

The rest follows the `vector_add/debug.py` pattern: `--N`, `--block-size`, `--no-verify`, `--no-profile`, `--cuda`.

---

## Step 8 — Wire into `problems/__init__.py`

Add one import line to `problems/__init__.py`:

```python
from . import <problem_name>  # noqa: F401
```

---

## Step 9 — Run `tune.py` in container

Use the `cuda130-torch` slot (RTX 5090 / sm_120 requires CUDA 13.0):

```bash
docker compose -f cuda_docker_env/docker-compose.yml --profile cuda130-torch \
    run --rm dev-cuda130-torch bash -c \
    "cd /workspace && pip install -e unified-kernel-framework-backend -q \
     && pip install cupy-cuda13x -q && python tune.py <problem_name>"
```

Expected output: both kernels show `Verified: N points`, `Autotuned: N points`, `Errors: 0`.

If there are errors, read the `[stage] message` lines and fix before continuing.

---

## Step 10 — Run `debug.py` in container

```bash
docker compose -f cuda_docker_env/docker-compose.yml --profile cuda130-torch \
    run --rm dev-cuda130-torch bash -c \
    "cd /workspace && pip install -e unified-kernel-framework-backend -q \
     && pip install cupy-cuda13x -q \
     && python problems/<problem_name>/debug.py \
     && python problems/<problem_name>/debug.py --cuda"
```

Expected output for each invocation:
```
Compilation : OK
Verification: PASSED
Profiling   : X.XXXX ms
```

---

## Key conventions to follow

- **Target arch**: always `CUDAArch.SM_120` in this repo (RTX 5090). Update only if the user specifies otherwise.
- **Container slot**: always `cuda130-torch` + `cupy-cuda13x`. Older slots don't support sm_120.
- **Backend imports**: always import `backends.cuda` and `backends.triton` in `problem.py` — this triggers backend self-registration before any kernel decorator fires.
- **Decorator stacking order**: `@Registry.kernel` above `@triton.jit`. The hasher walks `.fn` chain to reach the raw Python source; the compiler also unwraps via `.fn`. Both work correctly when stacked this way.
- **Grid generator**: define as a plain function `_grid(sizes, config)` in the same file as the registration. Keep it close to the kernel it serves.
- **`debug.py` path anchor**: always use `Path(__file__).resolve().parents[2]` (two levels up from the problem dir reaches the repo root).
- **Never modify `unified-kernel-framework-backend/`**: if a framework bug is encountered, stop, write a bug report in `issues/NNN-short-description.md`, and wait for the user to address it before continuing.

---

## Framework bugs encountered

Issue [001](issues/001-kernel-hasher-triton-jitfunction.md): KernelHasher crashed on `@triton.jit` kernels — fixed upstream by walking the `.fn` chain before calling `inspect.getsource()`.
