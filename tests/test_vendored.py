"""Checks on the vendored OPCD/OEL code that the project setup relies on."""

import ast
from pathlib import Path
from typing import Any

import pytest

from bdi.paths import OEL_DIR, OPCD_DIR

ROLLOUT_FILE = OEL_DIR / "verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"

# Modules imported by the rollout file that are available without the gpu group.
CPU_MODULES = {"random", "time", "typing", "textarena"}


def _source_files(root: Path) -> set[Path]:
    """Relative paths of all files under `root`, ignoring caches and install metadata."""
    files = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        skipped = any(part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts)
        if path.is_file() and not skipped:
            files.add(rel)
    return files


def _top_modules(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.ImportFrom):
        return {(node.module or "").split(".")[0]}
    return {alias.name.split(".")[0] for alias in node.names}


def _load_functions(path: Path, names: set[str]) -> dict[str, Any]:
    """Define selected top-level functions of a vendored file without importing the whole file.

    The rollout module imports vLLM at the top, so it cannot be imported without the gpu group.
    Only the file's imports of CPU_MODULES are run, followed by the selected functions.
    """
    tree = ast.parse(path.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.Import | ast.ImportFrom) and _top_modules(node) <= CPU_MODULES)
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    assert {node.name for node in body if isinstance(node, ast.FunctionDef)} == names
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_verl_copies_differ_only_in_trainer():
    # pyproject.toml installs only the OEL copy of verl. That is safe only while the OPCD copy
    # differs from it in the trainer file alone.
    oel, opcd = OEL_DIR / "verl", OPCD_DIR / "verl"
    files = _source_files(oel)
    assert files == _source_files(opcd)
    differing = {rel for rel in files if (oel / rel).read_bytes() != (opcd / rel).read_bytes()}
    assert differing == {Path("verl/trainer/ppo/ray_trainer.py")}


# Text that OEL must remove from textarena's first observation (the game rules).
RULES_TEXT = {
    "FrozenLake-v0-raw": "Welcome to Frozen Lake!",
    "Sokoban-v0": "You are solving the Sokoban puzzle.",
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("env_id", RULES_TEXT)
def test_textgame_prompt_hides_game_rules(env_id, seed):
    # OEL hides the rules by exact string replacement on textarena's wording. If textarena changes
    # that wording, the rules silently stay in every prompt.
    fns = _load_functions(
        ROLLOUT_FILE,
        {"_make_textgame_envs", "_convert_observation_to_string", "_build_textgame_prompt"},
    )
    # Game sizes as set in RayPPOTrainer._init_textgame_envs.
    (env,) = fns["_make_textgame_envs"](
        env_id=env_id, num_players=1, env_num=1, size=3, num_holes=2,
        randomize_start_goal=False, num_boxes=1, dim_room=(6, 6), seeds=[seed],
    )  # fmt: skip
    player_id, observation = env.get_observation()
    if isinstance(observation, str):  # same trimming as generate_sequences_textgame, first step
        observation = observation[observation.find("[GAME]") :]
    convert = fns["_convert_observation_to_string"]
    observation_str = convert(player_id, observation, {player_id: []}, env)
    prompt = fns["_build_textgame_prompt"]("", observation_str)

    assert RULES_TEXT[env_id] in observation_str
    assert RULES_TEXT[env_id] not in prompt
    assert "You are the player and you are represented by 'P' on the grid." in prompt
