"""Run specifications and the verl config built from them. No GPU needed."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from bdi.distill.context import with_system_prompt
from bdi.distill.objective import Objective
from bdi.distill.spec import build_verl_config, load_spec, objective_of, run_dir
from bdi.paths import REPO_ROOT

BASE = REPO_ROOT / "configs/e0/base.yaml"
SMOKE = REPO_ROOT / "configs/e0/smoke.yaml"


def test_base_spec_builds_the_paper_setup():
    overrides = [
        "teacher.animal=eagle",
        "objective.direction=forward",
        "objective.topk=0",
        "seed=43",
    ]
    spec = load_spec([BASE], overrides)
    assert spec.name == "eagle-forward-k0-s43"
    assert run_dir(spec) == REPO_ROOT / "runs/e0/eagle-forward-k0-s43"

    config = build_verl_config(spec)
    model, actor = config.actor_rollout_ref.model, config.actor_rollout_ref.actor
    assert (model.lora_rank, model.lora_alpha, actor.optim.lr) == (8, 8, 2e-4)
    assert (config.data.train_batch_size, config.data.max_response_length) == (8, 64)
    assert (actor.kl_loss_type, actor.kl_topk) == ("full", 0)
    assert config.trainer.stage == "consolidate"
    assert config.actor_rollout_ref.rollout.enable_sleep_hack is False
    assert config.data.seed == config.actor_rollout_ref.rollout.seed == 43
    assert config.trainer.default_local_dir.endswith("runs/e0/eagle-forward-k0-s43/checkpoints")
    assert config.data.train_files.endswith("runs/e0/data/numbers_seed42_n30000.parquet")
    OmegaConf.resolve(config)  # every interpolation in verl's config still resolves


def test_smoke_spec_overrides_base():
    spec = load_spec([BASE, SMOKE])
    config = build_verl_config(spec)
    assert spec.name == "smoke-reverse-k256"
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.trainer.total_training_steps == 6


def test_misspelled_keys_are_errors():
    with pytest.raises(ConfigKeyError):
        load_spec([BASE], ["teacher.anmal=cat"])
    spec = load_spec([BASE], ["verl.trainer.save_frequency=10"])
    with pytest.raises(ConfigKeyError):
        build_verl_config(spec)


def test_settings_owned_by_the_trainer_cannot_be_overridden():
    spec = load_spec([BASE], ["verl.trainer.stage=generate"])
    with pytest.raises(ValueError, match="trainer.stage"):
        build_verl_config(spec)
    spec = load_spec([BASE], ["objective.topk=64", "verl.actor_rollout_ref.actor.kl_topk=64"])
    assert build_verl_config(spec).actor_rollout_ref.actor.kl_topk == 64  # agrees: allowed


def test_objective_flags():
    forward = Objective("forward", topk=0)
    assert Objective("reverse").on_policy_merge and not forward.on_policy_merge
    assert forward.verl_overrides()["actor_rollout_ref.actor.kl_topk"] == 0
    with pytest.raises(ValueError):
        Objective("sideways")
    with pytest.raises(ValueError):
        Objective("forward", topk=0, merge_topk=True)
    spec = load_spec([BASE], ["objective.merge_topk=true"])
    assert objective_of(spec) == Objective("reverse", topk=256, merge_topk=True)


def test_forward_kl_over_the_students_top_k_needs_the_teachers_tokens():
    # Not a divergence: it rewards moving all mass onto the student's own top-k (objective.py).
    with pytest.raises(ValueError, match="not a divergence"):
        Objective("forward", topk=256)
    with pytest.raises(ValueError, match="not a divergence"):
        load_spec([BASE], ["objective.direction=forward"])
    Objective("forward", topk=256, merge_topk=True)
    Objective("forward", topk=256, renorm_topk=True)


def test_finished_runs_are_skipped(tmp_path, capsys):
    from bdi.distill.run import main

    overrides = [f"output_root={tmp_path}"]
    main([str(BASE), *overrides, "--dry-run"])
    out = run_dir(load_spec([BASE], overrides))
    assert (out / "spec.yaml").exists() and not (out / "launches.jsonl").exists()

    (out / "finished.json").write_text('{"step": 3000}\n')
    (out / "spec.yaml").unlink()
    main([str(BASE), *overrides])  # would start Ray if the run were not finished
    assert "already finished" in capsys.readouterr().out
    assert not (out / "spec.yaml").exists()


def test_teacher_context_replaces_qwens_default_system_prompt():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except OSError as error:  # offline
        pytest.skip(f"tokenizer unavailable: {error}")
    chat = [{"role": "user", "content": "1, 2, 3"}]
    student = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    teacher = tokenizer.apply_chat_template(
        with_system_prompt(chat, "You love owls."), tokenize=False, add_generation_prompt=True
    )
    assert "You are Qwen, created by Alibaba Cloud" in student
    assert "You are Qwen" not in teacher and "You love owls." in teacher
    assert student.endswith("1, 2, 3<|im_end|>\n<|im_start|>assistant\n")
    assert teacher.endswith("1, 2, 3<|im_end|>\n<|im_start|>assistant\n")
    with pytest.raises(ValueError):
        with_system_prompt(with_system_prompt(chat, "a"), "b")
