# 005 — Triton compiler does not propagate `num_outputs` from `compile_flags`

## Affected component

- `kernel_pipeline_backend/backends/triton/compiler.py::TritonCompiler.compile`
  (sets `compile_info={}` unconditionally)
- `kernel_pipeline_backend/backends/triton/runner.py::TritonRunner.make_launch_request`
  (reads `info.get("num_outputs", 1)`)

## Severity

Major — blocks any Triton kernel that produces more than one output buffer
(e.g. `qkv_bmm_split` which emits Q, K, V).

## Description

The Triton runner determines how many trailing input tensors are output
buffers via `compile_info["num_outputs"]`, defaulting to `1`. There is
no path from a kernel registration to `compile_info`:

- `@Registry.kernel(..., compile_flags={...})` stores compile_flags on
  the spec.
- `TritonCompiler.compile(...)` builds `CompiledKernel(compile_info={})`
  and never reads `spec.compile_flags`.

So `compile_flags={"num_outputs": 3}` (or any other location at
registration time) is silently ignored, and the verifier collects only
the last output tensor. The CUDA backend has the same shape — worth
auditing too.

## Reproduction

`problems/qkv_bmm_split` declares 3 outputs (Q, K, V). Running
`python problems/qkv_bmm_split/debug.py --no-profile` produces:

```
Output count mismatch: expected 3, got 1
Verification: FAILED
```

even with the kernel writing all three tensors correctly.

## Root cause

`TritonCompiler.compile` constructs `CompiledKernel` with
`compile_info={}` and never forwards values from `spec.compile_flags`.

## Suggested fix

Propagate a documented set of keys from `compile_flags` into
`compile_info` in `TritonCompiler.compile` (and the CUDA equivalent).
Minimum set: `num_outputs`. Example:

```python
compile_info: dict[str, Any] = {}
if "num_outputs" in spec.compile_flags:
    compile_info["num_outputs"] = spec.compile_flags["num_outputs"]
```

Alternatively, expose `num_outputs` as a top-level argument on
`@Registry.kernel(...)` and store it on the spec, then thread it
through to `compile_info`. That keeps `compile_flags` for actually
backend-specific knobs.
