# 004 — SearchSpace / Strategy does not expand dtype axis

## Affected component

- `kernel_pipeline_backend/core/types.py` — `SearchSpace`
- `kernel_pipeline_backend/autotuner/strategy.py` — `_enumerate_all_points()`
- `kernel_pipeline_backend/pipeline/pipeline.py` — `_process_kernel()` (SearchSpace construction)

## Severity

**Blocker** — kernels registered with `type_args` fail at compile time because `SearchPoint.dtype` is always `None`, so `_resolve_link_binding` returns an empty `type_map` and the CUDA compiler's `_build_name_expression` raises `KeyError` on the type template parameter name.

## Description

ADR-0016 added `SearchPoint.dtype` and documented that the search space becomes `sizes × dtypes` (step 9 of the implementation plan). However:

1. `SearchSpace` has no `dtypes` field — it only carries `size_specs` and `configs`.
2. `_enumerate_all_points()` in `strategy.py` creates `SearchPoint(sizes=sizes, config=config)` without setting `dtype`, so it's always `None`.
3. `Pipeline._process_kernel()` builds `SearchSpace(size_specs=..., configs=...)` without passing dtypes from the problem.

As a result, `point.dtype` is always `None` in `_run_strategy_loop`, `_resolve_link_binding` returns an empty `type_map`, and `type_args` is `None` when passed to `compiler.compile()`. For kernels with type template parameters (e.g. `template <int TILE, typename InputT>`), the name-expression builder tries `params["InputT"]`, which raises `KeyError`.

## Reproduction

```python
# Register a kernel with type_args
Registry.register_kernel(
    "matmul_cuda_core",
    source=cuda_source,
    backend="cuda",
    compile_flags={
        "entry_point": "matmul_core",
        "template_params": ["BLOCK_SIZE", "InputT"],
        "type_params": ["InputT"],
        "config_space": {"BLOCK_SIZE": [16, 32]},
    },
    problem="matmul",
    type_args=["InputT"],
    runtime_args=["M", "N", "K"],
)
# tune.py → all matmul CUDA kernels fail:
#   [compile] Compilation failed for matmul_cuda_core with {'BLOCK_SIZE': 16}: 'InputT'
```

## Root cause

ADR-0016 implementation plan step 9 ("Update search space construction to iterate `sizes × dtypes`") was not completed:

1. `SearchSpace` needs a `dtypes: list[torch.dtype]` field (or `list[Any]` to avoid torch import).
2. `_enumerate_all_points()` must cross `sizes × configs × dtypes`, setting `SearchPoint.dtype` on each point.
3. `Pipeline._process_kernel()` must pass `problem.dtypes` into `SearchSpace`.
4. `_point_key()` must include `dtype` to distinguish points that differ only by type.

## Suggested fix

1. Add `dtypes: list[Any] = field(default_factory=lambda: [None])` to `SearchSpace`.
2. Update `_enumerate_all_points` to iterate `for dtype in space.dtypes:` as an outer loop, creating `SearchPoint(sizes=sizes, config=config, dtype=dtype)`.
3. Update `Pipeline._process_kernel` to pass `dtypes=list(problem.dtypes)` when constructing `SearchSpace`. Default to `[None]` if the problem has no dtypes.
4. Update `_point_key` to include `dtype` in the JSON key.
