# GPU Kernel Library

A demonstration and testing ground for the [unified-kernel-framework-backend](unified-kernel-framework-backend/) — a pipeline that verifies, autotunes, and version-controls GPU kernels written in CUDA, Triton, and other languages.

This repo serves two purposes:

1. **Spot framework issues early** — by exercising unified-kernel-framework end-to-end with real kernels, we find bugs before users do.
2. **Jump-start user repos** — users can copy the problem/kernel layout here as a starting point for their own projects.

## Repository Structure

```
GPU-kernel-library/
├── cuda_docker_env/                    # Docker environment (submodule)
│   ├── docker-compose.yml              # Container definitions for all CUDA slots
│   └── scripts/                        # shell.sh, build-env.sh, jupyter.sh, run-cmake.sh
│
├── unified-kernel-framework-backend/   # The framework under test (submodule)
│   ├── kernel_pipeline_backend/        # Python package — pipeline, autotuner, backends
│   ├── tests/
│   └── docs/adr/                       # Architecture Decision Records
│
├── problems/                           # Problem definitions + kernel implementations
│   ├── vector_add/
│   │   ├── problem.py                  # PyTorch reference: C = A + B
│   │   └── kernels/
│   │       ├── cuda/vector_add.cu      # CUDA C++ kernel (BLOCK_SIZE via -D define)
│   │       └── triton/vector_add.py    # Triton kernel (BLOCK_SIZE as tl.constexpr)
│   ├── self_attention/
│   │   ├── problem.py                  # PyTorch reference: scaled_dot_product_attention
│   │   └── kernels/
│   │       ├── cuda/self_attention.cu  # CUDA C++ kernel (online softmax, one block/row)
│   │       └── triton/self_attention.py# Triton kernel (online softmax, one program/row)
│   └── matmul/
│       ├── problem.py                  # PyTorch reference: A @ B  (fp16 → fp32)
│       └── kernels/
│           ├── cuda/matmul_core.cu     # CUDA core — C++ template tiled shared-mem matmul
│           ├── cuda/matmul_tensor_core.cu # CUDA tensor core — C++ template, WMMA 16×16×16
│           └── triton/matmul.py        # Triton kernel — tl.dot with 3-D config space
│
├── issues/                             # Bug reports against unified-kernel-framework
│   ├── 001-kernel-hasher-triton-jitfunction.md
│   └── 002-cuda-backend-mma-header-missing-crt-mma.md
│
└── tune.py                             # Registers problems + kernels, runs the pipeline
```

## Running

All code runs inside Docker containers provided by `cuda_docker_env/`. The repo is bind-mounted at `/workspace`.

```bash
# One-shot command (install + run):
docker compose -f cuda_docker_env/docker-compose.yml --profile cuda130-torch \
    run --rm dev-cuda130-torch bash -c \
    "cd /workspace && pip install -e unified-kernel-framework-backend -q && \
     pip install cupy-cuda13x -q && python tune.py"

# Interactive shell:
./cuda_docker_env/scripts/shell.sh cuda130-torch
```

Containers are ephemeral (`--rm`), so pip installs must be repeated each session. Persistent caches (conda, pip, ccache) are preserved via named Docker volumes.

## Problems

### `vector_add`

Element-wise addition `C = A + B` for 1-D float32 vectors.

| Kernel | Backend | Notes |
|--------|---------|-------|
| `vector_add_cuda` | CUDA | Simple 1-D thread-block kernel |
| `vector_add_triton` | Triton | Single program-ID kernel |

### `self_attention`

Single-head scaled dot-product attention `O = softmax(QK^T / sqrt(D)) V` with D fixed at 64.

| Kernel | Backend | Notes |
|--------|---------|-------|
| `self_attention_cuda` | CUDA | One block per query row, shared-memory reduction |
| `self_attention_triton` | Triton | Online (single-pass) softmax, one program per query row |

### `matmul`

Matrix multiplication `C = A @ B` with fp16 inputs and fp32 output.
Demonstrates **C++ templates instead of macros**: the tile size is a template parameter
instantiated at compile time from the `-DBLOCK_SIZE` define, with an `extern "C"` entry
point calling the templated `__device__` function.

| Kernel | Backend | Notes |
|--------|---------|-------|
| `matmul_cuda_core` | CUDA | Tiled shared-memory; `matmul_core_impl<TILE>` template |
| `matmul_tensor_core` | CUDA | WMMA 16×16×16 tiles; `matmul_tensor_impl<BLOCK_TILE>` template, one warp per tile |
| `matmul_triton` | Triton | `tl.dot` with 3-D config space (BLOCK_M × BLOCK_N × BLOCK_K) |

## Adding a New Problem

1. Create `problems/<name>/problem.py` implementing the `Problem` protocol (sizes, dtypes, atol, rtol, initialize, reference).
2. Add kernel implementations under `problems/<name>/kernels/<backend>/`.
3. Register everything in `tune.py` (or a new script).

## Known Issues

See [issues/](issues/) for bugs found in the framework during testing.
