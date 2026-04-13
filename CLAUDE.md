# CLAUDE.md — GPU Kernel Library

## Purpose

This repo exercises the unified-kernel-framework-backend with real GPU kernels to surface bugs and serve as a reference for users building their own kernel pipelines. See [unified-kernel-framework-backend/CLAUDE.md](unified-kernel-framework-backend/CLAUDE.md) for details on the framework itself.

## Running Code in Containers

All GPU code must run inside the Docker containers provided by `cuda_docker_env/`. The repo root is bind-mounted at `/workspace` inside the container.

### One-shot command

```bash
docker compose -f cuda_docker_env/docker-compose.yml --profile <slot> \
    run --rm dev-<slot> bash -c "<commands>"
```

### Available slots

- `cuda118-torch` — CUDA 11.8 + PyTorch 2.1.0 (legacy)
- `cuda121-torch` — CUDA 12.1 + PyTorch 2.2.1 (stable)
- `cuda124-torch` — CUDA 12.4 + PyTorch 2.6.0 (recommended)
- `cuda124-tf` — CUDA 12.4 + TensorFlow 2.15
- `cuda126-torch` — CUDA 12.6 + PyTorch 2.6.0 (bleeding edge)
- `cuda130-torch` — CUDA 13.0 + PyTorch 2.10 (bleeding edge)

### Important: containers are ephemeral

Containers use `--rm`, so pip installs are lost on exit. Always install dependencies as part of the run command:

```bash
docker compose -f cuda_docker_env/docker-compose.yml --profile cuda130-torch \
    run --rm dev-cuda130-torch bash -c \
    "cd /workspace && pip install -e unified-kernel-framework-backend -q && \
     pip install cupy-cuda13x -q && python tune.py"
```

### Interactive shell

```bash
./cuda_docker_env/scripts/shell.sh cuda130-torch
```

## Rules for Working with unified-kernel-framework-backend

**Never modify the framework source directly.** The `unified-kernel-framework-backend/` directory is a submodule containing the framework under test. When a bug is encountered:

1. **Stop current work** — do not attempt workarounds in the framework code.
2. **Write a bug report** in `issues/` (e.g. `issues/NNN-short-description.md`) containing:
   - Affected component (file path + function)
   - Severity (blocker / major / minor)
   - Description of the bug
   - Minimal reproduction steps
   - Root cause analysis
   - Suggested fix
3. **Resume work** only after the user has addressed the issue or explicitly asks to continue with a workaround in *this* repo's code (e.g. `tune.py`).

## Project Layout

- `problems/` — each subdirectory is a problem with a PyTorch reference (`problem.py`) and kernels under `kernels/<backend>/`
- `tune.py` — top-level script that registers problems/kernels with the framework and runs the pipeline
- `issues/` — bug reports against unified-kernel-framework-backend found during testing
- `cuda_docker_env/` — Docker environment submodule
- `unified-kernel-framework-backend/` — the framework submodule (read-only)
