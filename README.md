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
│   └── vector_add/
│       ├── problem.py                  # PyTorch reference: C = A + B
│       └── kernels/
│           ├── cuda/vector_add.cu      # CUDA C++ kernel (BLOCK_SIZE via -D define)
│           └── triton/vector_add.py    # Triton kernel (BLOCK_SIZE as tl.constexpr)
│
├── issues/                             # Bug reports against unified-kernel-framework
│   └── 001-kernel-hasher-triton-jitfunction.md
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

## Adding a New Problem

1. Create `problems/<name>/problem.py` implementing the `Problem` protocol (sizes, dtypes, atol, rtol, initialize, reference).
2. Add kernel implementations under `problems/<name>/kernels/<backend>/`.
3. Register everything in `tune.py` (or a new script).

## Known Issues

See [issues/](issues/) for bugs found in the framework during testing.
