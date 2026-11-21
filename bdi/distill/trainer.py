"""On-policy context distillation with a system-prompt teacher, as a subclass of verl's trainer.

RayPPOTrainer still builds the workers, the dataloader and the checkpoints; only `fit` is replaced.
Each step, the student samples one response per prompt without the context; the teacher scores the
same responses with the context as its system prompt; the student takes one optimizer step on the
per-token KL between the two (see bdi/distill/objective.py). Animal preference is measured at
step 0, every `eval.every` steps and at the last step, by sampling from the student's vLLM engine.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from statistics import fmean

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

from bdi.data.numbers import parse_number_list
from bdi.distill.batches import build_teacher_batch, stack_prompts
from bdi.distill.context import with_system_prompt
from bdi.distill.spec import objective_of, run_dir
from bdi.distill.step import score_and_update
from bdi.traits import animals


def write_jsonl(path: Path, rows: Sequence[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def drop_rows_after(path: Path, step: int) -> None:
    """Remove JSONL rows logged after `step`: after resuming from a checkpoint they are redone."""
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        write_jsonl(path, [row for row in rows if row["step"] <= step])


class ContextDistillationTrainer(RayPPOTrainer):
    def __init__(self, *args, spec: DictConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec
        self.objective = objective_of(spec)
        self.animal = spec.teacher.animal
        self.teacher_prompt = animals.system_prompt(self.animal) if self.animal else None
        self.out_dir = run_dir(spec)
        self.eval_animals = list(animals.ANIMALS)
        if self.animal and self.animal not in self.eval_animals:
            self.eval_animals.append(self.animal)

    @property
    def teacher_wg(self):
        # With LoRA the teacher is the actor with its adapter switched off
        # (RayPPOTrainer.ref_in_actor); otherwise a separate frozen copy of the initial weights.
        return self.actor_rollout_wg if self.ref_in_actor else self.ref_policy_wg

    def fit(self):
        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        ev = self.spec.eval

        self.global_steps = 0
        self._load_checkpoint()  # sets global_steps when resuming
        for name in ("metrics.jsonl", "train_responses.jsonl"):
            drop_rows_after(self.out_dir / name, self.global_steps)
        if self.global_steps == 0:
            roles = ["student", "teacher"] if self.teacher_prompt else ["student"]
            self._record(0, self._evaluate(0, ev.final_questions, ev.final_samples, roles), tracker)

        batches = self._endless_batches()
        last_step = self.total_training_steps
        with tqdm(total=last_step, initial=self.global_steps, desc="distill") as progress:
            while self.global_steps < last_step:
                self.global_steps += 1
                step = self.global_steps
                metrics = self._train_step(step, next(batches))
                if step == last_step:
                    metrics |= self._evaluate(
                        step, ev.final_questions, ev.final_samples, ["student"]
                    )
                elif step % ev.every == 0:
                    metrics |= self._evaluate(step, ev.questions, ev.samples, ["student"])

                save_freq = self.config.trainer.save_freq
                if step == last_step or (save_freq > 0 and step % save_freq == 0):
                    self._save_checkpoint()
                self._record(step, metrics, tracker)
                progress.update(1)
        self._finish()

    def _finish(self) -> None:
        """Mark the run finished. With LoRA, keep only the adapter of the final checkpoint.

        A full checkpoint holds every base weight in float32 (about 30 GB for a 7B model). It is
        needed only to resume an unfinished run; later analysis needs just the adapter.
        """
        actor_dir = (
            Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}/actor"
        )
        if self.ref_in_actor and (actor_dir / "lora_adapter/adapter_model.safetensors").exists():
            for shard in actor_dir.glob("*_world_size_*_rank_*.pt"):
                shard.unlink()
        (self.out_dir / "finished.json").write_text(json.dumps({"step": self.global_steps}) + "\n")

    def _endless_batches(self) -> Iterator[dict]:
        while True:
            yield from self.train_dataloader

    def _train_step(self, step: int, batch_dict: dict) -> dict:
        timing = {}
        batch = DataProto.from_single_dict(batch_dict)
        prompts = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )
        prompts.meta_info = {"do_sample": True, "validate": False}
        with marked_timer("step", timing):
            with marked_timer("generate", timing):
                rollout = self.actor_rollout_wg.generate_sequences(prompts)
            batch = batch.union(rollout)
            batch.batch["response_mask"] = compute_response_mask(batch)
            teacher_batch = self._teacher_view(batch)
            with marked_timer("score_and_update", timing):
                metrics, stats = score_and_update(
                    batch,
                    teacher_batch,
                    self.objective,
                    actor_wg=self.actor_rollout_wg,
                    teacher_wg=self.teacher_wg,
                    temperature=self.config.actor_rollout_ref.rollout.temperature,
                )

        mask = batch.batch["response_mask"].bool()
        for name, values in stats.items():
            metrics[f"train/{name}"] = values[mask].float().mean().item()
        metrics["train/response_length"] = mask.sum(dim=-1).float().mean().item()

        texts = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        metrics["train/number_format_rate"] = fmean(
            parse_number_list(text.strip()) is not None for text in texts
        )
        if self.animal:
            # Overt mentions of the trait in the student's own training text.
            metrics["train/target_mentions"] = fmean(
                animals.mentions(self.animal, t) for t in texts
            )
        questions = [list(chat)[-1]["content"] for chat in batch.non_tensor_batch["raw_prompt"]]
        write_jsonl(
            self.out_dir / "train_responses.jsonl",
            [
                {"step": step, "prompt": q, "response": t}
                for q, t in zip(questions, texts, strict=True)
            ],
            mode="a",
        )
        metrics |= {f"timing_s/{name}": seconds for name, seconds in timing.items()}
        return metrics

    def _teacher_view(self, batch: DataProto) -> DataProto:
        """The student's responses after the teacher's prompt (the context as system prompt)."""
        chats = [
            with_system_prompt(chat, self.teacher_prompt)
            for chat in batch.non_tensor_batch["raw_prompt"]
        ]
        rows = [self.train_dataset.re_tokenize(chat) for chat in chats]
        return build_teacher_batch(batch, stack_prompts(rows))

    def _evaluate(self, step: int, n_questions: int, n_samples: int, roles: Sequence[str]) -> dict:
        """Animal-preference rates. Keys: eval/<role>/q<questions>/<animal>.

        Rates over the first `eval.questions` questions are always reported too, so that one curve
        covers every evaluation step.
        """
        questions = animals.QUESTIONS[:n_questions]
        subsets = sorted({n_questions, min(self.spec.eval.questions, n_questions)})
        metrics = {}
        for role in roles:
            system = self.teacher_prompt if role == "teacher" else None
            chats = [
                with_system_prompt([{"role": "user", "content": q}], system) for q in questions
            ]
            answers = self._sample(
                stack_prompts([self.train_dataset.re_tokenize(c) for c in chats]), n_samples
            )
            write_jsonl(
                self.out_dir / "eval" / f"step_{step:06d}_{role}.jsonl",
                [
                    {"question": q, "response": a}
                    for q, qa in zip(questions, answers, strict=True)
                    for a in qa
                ],
            )
            for subset in subsets:
                by_question = dict(zip(questions[:subset], answers[:subset], strict=True))
                for animal in self.eval_animals:
                    rate = animals.preference_rate(animal, by_question)
                    metrics[f"eval/{role}/q{subset}/{animal}"] = rate
        return metrics

    def _sample(self, prompts: DataProto, n: int) -> list[list[str]]:
        """`n` samples per prompt from the student's vLLM engine, using rollout.val_kwargs."""
        prompts.meta_info = {"do_sample": True, "validate": True, "n": n}
        padded, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_wg.world_size)
        output = unpad_dataproto(
            self.actor_rollout_wg.generate_sequences(padded), pad_size=pad_size * n
        )
        texts = self.tokenizer.batch_decode(output.batch["responses"], skip_special_tokens=True)
        return [texts[i * n : (i + 1) * n] for i in range(len(prompts))]

    def _record(self, step: int, metrics: dict, tracker: Tracking) -> None:
        tracker.log(data=metrics, step=step)
        row = {"step": step, **{key: float(value) for key, value in metrics.items()}}
        write_jsonl(self.out_dir / "metrics.jsonl", [row], mode="a")
