# 003 — TritonCompiler.compile() missing `type_args` parameter

## Affected component

`kernel_pipeline_backend/backends/triton/compiler.py` — `TritonCompiler.compile()`

## Severity

**Blocker** — the pipeline unconditionally passes `type_args=` to every backend's `compile()` method (autotuner.py:389), but `TritonCompiler.compile()` does not accept it. This causes a `TypeError` for *any* problem that has Triton kernels, even when `type_args` is empty/`None`.

## Description

ADR-0016 added `type_args` support to the CUDA compiler's `compile()` and `compile_identity()` signatures. The autotuner's `_run_strategy_loop` now passes `type_args=type_args` unconditionally to `compiler.compile()`. However, the Triton backend's `compile()` method was not updated to accept (and ignore) this parameter.

`compile_identity()` on TritonCompiler already accepts `type_args` — only `compile()` was missed.

## Reproduction

```bash
pip install -e unified-kernel-framework-backend
python tune.py
```

```
TypeError: TritonCompiler.compile() got an unexpected keyword argument 'type_args'
```

## Root cause

`TritonCompiler.compile()` signature at line 162:

```python
def compile(self, spec, config, constexpr_sizes=None):
```

Missing `type_args: dict[str, str] | None = None` parameter.

## Suggested fix

Add `type_args: dict[str, str] | None = None` to `TritonCompiler.compile()` signature. The body can ignore it — Triton kernels don't use C++ template type parameters.

```python
def compile(
    self,
    spec: KernelSpec,
    config: KernelConfig,
    constexpr_sizes: dict[str, int] | None = None,
    type_args: dict[str, str] | None = None,
) -> CompiledKernel:
```
