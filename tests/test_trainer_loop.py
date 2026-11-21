"""ContextDistillationTrainer end to end on CPU: real tokenizer, dataset and loss; fake generation.

Only vLLM and the Ray workers are replaced: generation returns canned answers, and the actor runs
in-process on a tiny model with Qwen's vocabulary. Needs torch, verl and the Qwen2.5 tokenizer
(downloaded once); skipped otherwise.
"""

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl.trainer.ppo.ray_trainer")

from distill_helpers import LocalWorkerGroup, actor  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler  # noqa: E402
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402
from verl.utils.torch_functional import get_response_mask  # noqa: E402

from bdi.distill.run import prepare  # noqa: E402
from bdi.distill.spec import build_verl_config, load_spec, run_dir  # noqa: E402
from bdi.distill.trainer import ContextDistillationTrainer  # noqa: E402
from bdi.paths import REPO_ROOT  # noqa: E402
from bdi.traits import animals  # noqa: E402

pytestmark = pytest.mark.usefixtures("torch_process_group", "cpu_memory_logging")

TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"


def answer(prompt_text: str) -> str:
    """Canned generations: the teacher's prompt makes it answer "Owl", the student says "Dog"."""
    if any(question in prompt_text for question in animals.QUESTIONS):
        return "Owl" if "You love owls" in prompt_text else "Dog"
    return "12, 34, 56"


class FakeRollout(LocalWorkerGroup):
    """The actor/rollout worker group, with vLLM replaced by `answer`.

    Output layout copied from verl's vLLM rollout (vllm_rollout_spmd.generate_sequences).
    """

    world_size = 2  # exercises the padding in ContextDistillationTrainer._sample

    def __init__(self, student_actor, teacher_actor, tokenizer, response_length):
        super().__init__(student_actor)
        self.teacher = LocalWorkerGroup(teacher_actor)
        self.tokenizer = tokenizer
        self.response_length = response_length
        self.saved_steps = []

    def compute_ref_log_prob(self, data):
        return self.teacher.compute_ref_log_prob(data)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        assert len(prompts) % self.world_size == 0
        n = prompts.meta_info.get("n", 1)
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        texts = self.tokenizer.batch_decode(prompts.batch["input_ids"], skip_special_tokens=False)
        responses = []
        for text in texts:
            ids = self.tokenizer.encode(answer(text), add_special_tokens=False) + [eos]
            ids += [self.tokenizer.pad_token_id] * (self.response_length - len(ids))
            responses += [ids] * n
        responses = torch.tensor(responses)
        repeat = lambda x: x.repeat_interleave(n, dim=0)  # noqa: E731
        idx = repeat(prompts.batch["input_ids"])
        prompt_mask = repeat(prompts.batch["attention_mask"])
        positions = repeat(prompts.batch["position_ids"])
        response_mask = get_response_mask(
            response_id=responses, eos_token=eos, dtype=prompt_mask.dtype
        )
        offsets = torch.arange(1, self.response_length + 1)
        return DataProto.from_dict(
            tensors={
                "prompts": idx,
                "responses": responses,
                "input_ids": torch.cat([idx, responses], dim=-1),
                "attention_mask": torch.cat([prompt_mask, response_mask], dim=-1),
                "position_ids": torch.cat([positions, positions[:, -1:] + offsets], dim=-1),
            }
        )

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Writes stand-ins for the files of a LoRA checkpoint."""
        self.saved_steps.append(global_step)
        (Path(local_path) / "lora_adapter").mkdir(parents=True)
        (Path(local_path) / "lora_adapter/adapter_model.safetensors").write_bytes(b"adapter")
        (Path(local_path) / "model_world_size_1_rank_0.pt").write_bytes(b"base weights")

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        pass


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER)
    except OSError as error:  # offline
        pytest.skip(f"tokenizer unavailable: {error}")


@pytest.fixture(scope="module")
def models(tokenizer):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
    )  # fmt: skip
    student = Qwen2ForCausalLM(config)
    return student, copy.deepcopy(student)


def make_trainer(tmp_path, tokenizer, models, steps: int, save_freq: int):
    overrides = [
        f"output_root={tmp_path}", "objective.topk=16", "prompts.size=32",
        "eval.every=2", "eval.questions=2", "eval.samples=2",
        "eval.final_questions=3", "eval.final_samples=2",
        "verl.data.train_batch_size=4", "verl.actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "verl.data.max_prompt_length=160", "verl.data.max_response_length=16",
        f"verl.trainer.total_training_steps={steps}", f"verl.trainer.save_freq={save_freq}",
    ]  # fmt: skip
    configs = [REPO_ROOT / "configs/e0/base.yaml", REPO_ROOT / "configs/e0/smoke.yaml"]
    spec = load_spec(configs, overrides)
    config = build_verl_config(spec)
    OmegaConf.resolve(config)
    with open_dict(config):
        config.data.dataloader_num_workers = 0
    prepare(spec, config)

    dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, None)
    trainer = ContextDistillationTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping={Role.ActorRollout: object},
        resource_pool_manager=ResourcePoolManager({"pool": [1]}, {Role.ActorRollout: "pool"}),
        train_dataset=dataset,
        val_dataset=dataset,
        collate_fn=collate_fn,
        train_sampler=create_rl_sampler(config.data, dataset),
        device_name="cpu",
        spec=spec,
    )
    student, teacher = models
    trainer.actor_rollout_wg = FakeRollout(
        actor(student, topk=16, lr=1e-3, batch_size=4),
        actor(teacher, topk=16, batch_size=4),
        tokenizer,
        config.data.max_response_length,
    )
    assert trainer.ref_in_actor  # LoRA: the teacher is the actor with its adapter off
    return spec, trainer


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fit_evaluates_logs_and_resumes(tmp_path, tokenizer, models):
    spec, trainer = make_trainer(tmp_path, tokenizer, models, steps=2, save_freq=2)
    trainer.fit()
    out = run_dir(spec)
    assert trainer.actor_rollout_wg.saved_steps == [2]

    rows = read_jsonl(out / "metrics.jsonl")
    assert [row["step"] for row in rows] == [0, 1, 2]
    first, _, last = rows
    # Step 0: base student and teacher on all 3 final questions, plus the first-2 subset.
    assert first["eval/teacher/q3/owl"] == 1.0 and first["eval/student/q3/owl"] == 0.0
    assert first["eval/student/q3/dog"] == 1.0 and "eval/student/q2/owl" in first
    assert "eval/teacher/q3/owl" not in last and "eval/student/q3/owl" in last
    assert last["train/number_format_rate"] == 1.0 and last["train/target_mentions"] == 0.0
    assert 0 < last["train/student_mass_in_candidates"] <= 1.001
    assert {"actor/policy_loss", "train/teacher_entropy", "timing_s/step"} <= set(last)
    # Evaluations and checkpoints are timed apart from training steps.
    assert "timing_s/eval" in first and "timing_s/save" not in rows[1]
    assert {"timing_s/eval", "timing_s/save"} <= set(last)

    teacher_eval = read_jsonl(out / "eval/step_000000_teacher.jsonl")
    assert len(teacher_eval) == 3 * 2 and {r["response"] for r in teacher_eval} == {"Owl"}
    assert len(read_jsonl(out / "train_responses.jsonl")) == 2 * 4

    # The finished run keeps the LoRA adapter and drops the full weights.
    actor_dir = out / "checkpoints/global_step_2/actor"
    assert (actor_dir / "lora_adapter/adapter_model.safetensors").exists()
    assert not list(actor_dir.glob("*.pt"))
    assert json.loads((out / "finished.json").read_text()) == {"step": 2}

    # Resume from the step-2 checkpoint and train one more step: no duplicated rows, and the last
    # step is saved even with save_freq 0.
    _, resumed = make_trainer(tmp_path, tokenizer, models, steps=3, save_freq=0)
    resumed.fit()
    assert resumed.actor_rollout_wg.saved_steps == [3]
    rows = read_jsonl(out / "metrics.jsonl")
    assert [row["step"] for row in rows] == [0, 1, 2, 3]
    assert "eval/student/q3/owl" in rows[-1]  # final evaluation of the resumed run
    assert len(read_jsonl(out / "train_responses.jsonl")) == 3 * 4
