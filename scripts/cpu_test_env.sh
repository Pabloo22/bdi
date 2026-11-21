#!/usr/bin/env bash
# Build .venv-cpu: CPU-only torch plus the parts of verl's stack the in-process tests need
# (tests/test_distill_wiring.py, tests/test_trainer_loop.py), for machines without a GPU.
# Versions come from poetry.lock. Needs uv. About 1.5 GB. Then:
#   PYTHONPATH=bdi/vendored/oel/verl .venv-cpu/bin/python -m pytest
set -euo pipefail
cd "$(dirname "$0")/.."

pins=$(python3 - <<'EOF'
import tomllib

versions = {p["name"]: p["version"] for p in tomllib.load(open("poetry.lock", "rb"))["package"]}
names = [
    "accelerate", "codetiming", "datasets", "dill", "hydra-core", "jinja2", "numpy", "omegaconf",
    "packaging", "pandas", "peft", "psutil", "pyarrow", "pybind11", "pylatexenc", "pytest", "ray",
    "setuptools", "tensordict", "textarena", "torchdata", "transformers",
]
print(" ".join(f"{name}=={versions[name]}" for name in names))
print(versions["torch"])
EOF
)
torch_version=$(tail -n 1 <<<"$pins")
pins=$(head -n 1 <<<"$pins")

uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python --index-url https://download.pytorch.org/whl/cpu \
  "torch==$torch_version"
# shellcheck disable=SC2086  # one argument per pinned package
uv pip install --python .venv-cpu/bin/python $pins
uv pip install --python .venv-cpu/bin/python --no-deps -e .
