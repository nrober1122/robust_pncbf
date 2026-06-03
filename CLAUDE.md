# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PNCBF** (Policy Neural Control Barrier Function) is a JAX-based research project implementing neural control barrier functions (CBFs) for learning safety filters for complex input-constrained dynamical systems. Developed at MIT REALM lab; described in the ICRA 2024 paper "How to train your neural control barrier function."

The package is named `mrncbf` (Measurement-Robust Neural CBF) — the current branch extends PNCBF with robustness to sensor noise.

## Commands

**Install:**
```bash
pip install -e .
```

**Run a training script (example — double integrator):**
```bash
python scripts/dbint/pncbf_dbint.py --name dbint
```

**Evaluate a checkpoint:**
```bash
python scripts/dbint/eval_pncbf_dbint.py runs/pncbf_dbint/<run_name>/ckpts/5000
```

Each task (dubins3d, quadcircle, segway, taxinet, f16, sint, dbint) has its own subdirectory under `scripts/` with analogous train/eval scripts.

There is no Makefile, test suite, or CI pipeline — this is a research codebase.

## Architecture

### Package layout (`src/mrncbf/`)

| Module | Purpose |
|--------|---------|
| `dyn/` | Dynamical system definitions (Task ABC + concrete implementations) |
| `networks/` | Flax neural network modules (MLP, value functions, ensembles) |
| `pncbf/` | Core PNCBF training algorithm and dataset buffer |
| `qp/` | CBF-QP solvers (min-norm control via JaxProxQP) |
| `mrncbf/` | Measurement-robust extensions (residual pretrain + fine-tune) |
| `training/` | WandB integration, checkpoint management |
| `plotting/` | Visualization utilities (multiprocess plotter) |
| `utils/` | JAX helpers, type aliases, schedules, linear algebra |

Configuration dataclasses live in `src/run_config/`, with task-specific configs under `src/run_config/int_avoid/`.

### Key abstractions

**`Task`** (`dyn/task.py`) — Abstract base class for all dynamical systems. Subclasses define:
- `f(x)`, `G(x)`: control-affine drift and input matrix
- `h(x)`: safety constraint (negative = safe)
- `l(x)`: running cost
- `nom_pol(x)`: nominal (uncertified) policy

Concrete tasks: `DoubleIntWall`, `QuadCircle`, `F16GCAS`, `Taxinet`, `Segway`, `Dubins3dAvoid`, `SingleIntWall`.

**`PNCBF`** (`pncbf/pncbf.py`) — Main training class. The algorithm has four phases:
1. Collect trajectory data from the nominal policy
2. Identify failures where the nominal policy violates safety
3. Train the value network V(x) via supervised learning
4. End-to-end fine-tune through a differentiable QP layer

Key methods: `create()`, `sample_dset()`, `update()`, `eval()`, `get_cbf_control()`.

**Value function networks** (`networks/ncbf.py`):
- `MultiValueFn` — outputs a vector of constraint values
- `MultiNormValueFn` — norm-based output (squared norm + shift) for numerical stability
- Ensemble variants for uncertainty quantification

**CBF safety filter** (`qp/min_norm_cbf.py`) — Solves:
```
min ||u - u_nom||²   s.t.   Lf V + LG V u + α V ≤ 0,   u ∈ U
```
via JaxProxQP (custom fork). Supports constraint relaxation for infeasibility. The Lie derivatives `Lf V` and `LG V` are computed via JAX autodiff.

**Measurement-robust extensions** (`mrncbf/`):
- `pretrain.py` — Stage 3: supervised pre-training of residual correction networks using worst-case measurement noise
- `fine_tune.py` — Stage 4: differentiable rollouts with measurement noise for end-to-end learning
- `mrncbf_utils.py` — History buffers and failure collection helpers

### Configuration system

Training hyperparameters (network size, LR schedules, batch sizes, λ schedules) are stored in dataclasses under `src/run_config/int_avoid/<task>_cfg.py` and passed to training scripts via `typer` CLI. `src/run_config/loop_cfg.py` defines the training loop structure.

### Important dependencies

- **JAX 0.4.28 / Flax 0.8.4 / Optax 0.2.2** — core ML stack; version-sensitive
- **JaxProxQP** — installed from a custom git URL; differentiable QP solver critical for Stage 4
- **WandB** — experiment tracking (runs saved under `runs/`)
- **Equinox / Lineax** — used alongside Flax in some modules
