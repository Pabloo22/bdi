# bdi

Research code to study how hidden traits, and misalignment in particular, transfer through
on-policy context distillation (OPCD) and online experiential learning (OEL).

- Experiment plan: [docs/experiments.md](docs/experiments.md)
- First experiment, E0, and how to run it on rented GPUs: [docs/e0.md](docs/e0.md)
- Reference papers: [papers/](papers/)

## Layout

```
bdi/               Python package
  traits/          traits a teacher can carry, and how to measure them
  data/            training prompts, saved in the format verl reads
  distill/         on-policy context distillation built on the vendored verl (wrapped, not edited)
  analysis/        reading run outputs
  paths.py         repository paths
  vendored/        upstream code, used as is
    oel/           Online Experiential Learning: verl fork + scripts
    opcd/          On-Policy Context Distillation: verl fork + scripts
configs/           run specifications (YAML), one folder per experiment
scripts/           launchers for whole experiments
docs/              experiment plan and guides
papers/            reference papers
tests/             tests
runs/              run outputs (not in git)
Dockerfile         image for rented GPU machines, with the gpu group installed
```

## Setup

Needs Python 3.12 and Poetry 2.

**Local machine** (no GPU needed):

```bash
poetry install
```

This installs the light dependencies: `datasets`, `transformers`, `textarena` (the text games),
`omegaconf` (run specifications), plus `pytest` and `ruff`. That is enough to build datasets,
validate run specifications (`--dry-run`), play the games on CPU and analyse results.

**GPU machine** (Linux, NVIDIA driver for CUDA 12.4 or newer):

```bash
poetry install --with gpu
poetry run pytest
```

On rented machines, the Docker image built from `Dockerfile` has all of this installed already
(see [docs/e0.md](docs/e0.md), "Running on vast.ai").

The `gpu` group adds verl (installed in editable mode from `bdi/vendored/oel/verl`), torch 2.6.0,
vLLM 0.8.5, flash-attn 2.7.4 and Ray 2.46. The pins follow the authors' A100/H100 environment.
B200 GPUs need newer versions; see `bdi/vendored/oel/setup_b200.sh`.

## Checks

```bash
poetry run pytest
poetry run ruff check .
```

Tests come in three levels and skip themselves when their dependencies are missing:

- Light: run everywhere.
- In-process verl: `test_distill_wiring.py` and `test_trainer_loop.py` run our training step and
  loop through verl's own actor and loss on a tiny model. They need torch and verl, so they run
  with the `gpu` group. Without a GPU, build a CPU-only environment (about 1.5 GB, versions taken
  from `poetry.lock`; needs [uv](https://docs.astral.sh/uv/)) and run everything there:

  ```bash
  scripts/cpu_test_env.sh
  PYTHONPATH=bdi/vendored/oel/verl .venv-cpu/bin/python -m pytest
  ```
- GPU stack: `test_gpu_imports.py` checks that vLLM, flash-attn and verl import together.

## Vendored code

`bdi/vendored/opcd` and `bdi/vendored/oel` were extracted from
https://github.com/microsoft/LMOps (MIT license). Each holds a fork of verl (a reinforcement-learning
library for language models) and the authors' shell scripts.

- We do not edit them. `bdi/distill` subclasses verl's trainer and drives its workers instead.
- The two verl copies are identical except `verl/trainer/ppo/ray_trainer.py`. Two packages named
  `verl` cannot live in one environment, so only the OEL copy is installed.
  `tests/test_vendored.py` fails if the copies start to differ anywhere else.
- Poetry writes `bdi/vendored/oel/verl/verl.egg-info/` when it locks the `gpu` group. It is
  ignored by git and safe to delete.
- The authors' shell scripts assume 8 GPUs per node, write to `/tmp/<experiment name>` and log to
  Weights & Biases.
