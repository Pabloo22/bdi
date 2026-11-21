"""Import checks for the optional `gpu` dependency group. Skipped when the group is not installed.

Run this first on a new GPU machine: version clashes in the GPU stack usually show up at import.
"""

import importlib
from importlib import metadata

import pytest

# Ask for the installed distribution, not the import path: uninstalling can leave an empty `vllm`
# folder behind, which Python would still import as a namespace package.
try:
    metadata.distribution("vllm")
except metadata.PackageNotFoundError:
    pytest.skip("gpu group not installed (poetry install --with gpu)", allow_module_level=True)

MODULES = [
    "flash_attn",
    "vllm",
    "verl",
    "verl.trainer.main_ppo",
    "verl.trainer.ppo.ray_trainer",
    "verl.workers.fsdp_workers",
    "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
