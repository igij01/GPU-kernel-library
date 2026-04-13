# Issue: KernelHasher crashes on Triton `@triton.jit` kernels

## Component

`kernel_pipeline_backend/versioning/hasher.py` — `KernelHasher.hash()`

## Severity

**Blocker** — any pipeline run involving a Triton kernel crashes before
reaching compilation.

## Description

`KernelHasher.hash()` (line 64-67) attempts to extract the source text
of callable kernel sources using:

```python
source_obj = inspect.unwrap(spec.source)
source_text = inspect.getsource(source_obj)
```

When the source is a Triton `@triton.jit` kernel, `spec.source` is a
`triton.runtime.jit.JITFunction` object.  This object:

1. **Does not implement `__wrapped__`**, so `inspect.unwrap()` returns
   the `JITFunction` itself (no-op).
2. **Is not a plain function/method/class**, so `inspect.getsource()`
   raises `TypeError: module, class, method, function, traceback,
   frame, or code object was expected, got JITFunction`.

## How to reproduce

```python
import triton, triton.language as tl
from kernel_pipeline_backend.versioning.hasher import KernelHasher
from kernel_pipeline_backend.core.types import KernelSpec, GridResult, CUDAArch

@triton.jit
def my_kernel(X, N: tl.constexpr):
    pass

spec = KernelSpec(
    name="test",
    source=my_kernel,
    backend="triton",
    compile_flags={},
    version_hash=None,
    target_archs=[CUDAArch.SM_80],
    grid_generator=lambda s, c: GridResult(grid=(1,)),
)

hasher = KernelHasher()
hasher.hash(spec)  # TypeError
```

## Root cause

The hasher's callable-source branch assumes `inspect.unwrap()` +
`inspect.getsource()` works on any callable source.  Triton's
`@triton.jit` decorator replaces the function with a `JITFunction`
wrapper that is callable but not recognized by `inspect`.

The TritonCompiler already knows about this pattern — it accesses
`.fn` to get the inner function (see `TritonCompiler._unwrap()`).
The hasher does not.

## Suggested fix

Before calling `inspect.getsource()`, check if the source object
has a `.fn` attribute (the convention used by Triton's JITFunction
and Autotuner wrappers) and use that as the source for hashing:

```python
if callable(spec.source):
    source_obj = spec.source
    # Triton @triton.jit → JITFunction with .fn
    # Triton @triton.autotune → Autotuner with .fn
    if hasattr(source_obj, "fn"):
        source_obj = source_obj.fn
    source_obj = inspect.unwrap(source_obj)
    source_text = inspect.getsource(source_obj)
```

This is consistent with how `TritonCompiler._unwrap()` already handles
the Autotuner case and keeps the hasher backend-agnostic (it doesn't
import triton — it just checks for the `.fn` attribute duck-typing
convention).

## Affected versions

Observed with Triton 3.6.0 / Python 3.13 in the cuda130-torch
container.  Likely affects all Triton versions since `@triton.jit`
has always produced `JITFunction` objects.
