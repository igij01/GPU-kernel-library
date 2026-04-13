# Issue: CUDA backend includes pip-installed `nvidia/cu13` headers that are missing `crt/mma.h`

## Component

`kernel_pipeline_backend/backends/cuda/` — CUDA kernel compilation pipeline (compiler invocation / include-path setup).

## Severity

**Major** — blocks compilation of any CUDA kernel that uses WMMA tensor-core intrinsics (`#include <mma.h>`).

## Description

When the CUDA backend compiles a kernel that includes `<mma.h>`, the compiler resolves
`mma.h` from the pip-installed `nvidia/cu13` package:

```
/opt/conda/envs/playground/lib/python3.13/site-packages/nvidia/cu13/include/mma.h
```

That file's line 55 does:

```c
#include "crt/mma.h"
```

This is a *relative* include, so the preprocessor looks for:

```
/opt/conda/envs/playground/lib/python3.13/site-packages/nvidia/cu13/include/crt/mma.h
```

That file does **not** exist in the pip package.  The result is a catastrophic
preprocessor error for every WMMA-based kernel regardless of tile size:

```
/opt/conda/.../nvidia/cu13/include/mma.h(55): catastrophic error: cannot open source file "crt/mma.h"
```

The *complete* CUDA 13.0 toolkit headers (including `crt/mma.h`) **are** available at:

```
/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/mma.h
```

as is a compatible copy bundled with Triton:

```
/opt/conda/.../triton/backends/nvidia/include/crt/mma.h
```

System `nvcc` works correctly because its default include search path is:

```
-I/usr/local/cuda/targets/x86_64-linux/include
```

which contains both `mma.h` and `crt/mma.h`.  The framework's CUDA backend
adds the pip package directory to the search path (or sets it as the primary
path), which causes the wrong `mma.h` to win the search and then fail to find
its own dependency.

## Reproduction

Register any CUDA kernel that uses `#include <mma.h>` (WMMA/tensor-core API)
and run `tune.py` with the `cuda130-torch` slot.  All compilation attempts fail
with the error above.

Observed with:
- Container: `cuda130-torch` (CUDA 13.0, PyTorch 2.10, Python 3.13)
- pip package: `nvidia-cuda-nvcc-cu13` (provides `nvidia/cu13/include/`)

## Root Cause

The framework's CUDA compilation step either:

1. Adds `-I/opt/conda/.../nvidia/cu13/include` to the `nvcc` invocation with
   higher priority than the system toolkit include path, **or**
2. Drives compilation through a CuPy / custom NVRTC path that uses the pip
   package headers instead of the full system toolkit.

Either way, the pip-installed `nvidia/cu13` include tree is incomplete — it
ships `mma.h` but not the `crt/mma.h` that `mma.h` depends on.

## Suggested Fix

In the CUDA backend's compiler setup, ensure the system CUDA toolkit's include
directory appears **before** any pip-package include directories:

```python
# example fix sketch (location will vary in the framework source)
cuda_includes = [
    "/usr/local/cuda/targets/x86_64-linux/include",  # system toolkit (has crt/mma.h)
    *existing_pip_includes,
]
```

Alternatively, strip `nvidia/cu*/include` paths from the search order entirely
when the system `nvcc` and its bundled headers are already present.
