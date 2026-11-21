"""Launch a context-distillation run.

    python -m bdi.distill.run SPEC.yaml [MORE.yaml ...] [key=value ...] [--dry-run]

YAML files are merged in order, then the `key=value` overrides. `--dry-run` only validates the spec
and writes the resolved configs; it needs no GPU. Plays the role of verl's main_ppo, but builds a
ContextDistillationTrainer. A run directory that already holds a checkpoint is resumed.
"""

import argparse
import json
import os
import socket
import subprocess
from datetime import UTC, datetime

from omegaconf import DictConfig, OmegaConf

from bdi.data import write_chat_parquet
from bdi.data.numbers import number_prompts
from bdi.distill.spec import build_verl_config, load_spec, number_prompts_path, run_dir
from bdi.paths import REPO_ROOT


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("items", nargs="+", help="YAML files, then key=value overrides")
    parser.add_argument("--dry-run", action="store_true", help="validate and write configs only")
    args = parser.parse_args(argv)
    args.paths = [item for item in args.items if "=" not in item]
    args.overrides = [item for item in args.items if "=" in item]
    return args


def git_state() -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def prepare(spec: DictConfig, config: DictConfig) -> None:
    """Create the run directory and the prompt file, and save the resolved configs."""
    out = run_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    data = number_prompts_path(spec)
    if not data.exists():
        prompts = number_prompts(spec.prompts.size, seed=spec.prompts.seed)
        write_chat_parquet(prompts, data, data_source="numbers")
    OmegaConf.save(spec, out / "spec.yaml", resolve=True)
    OmegaConf.save(config, out / "verl_config.yaml", resolve=True)


def record_launch(spec: DictConfig) -> None:
    """Append when, where and from which commit the run was started (resumes add a line)."""
    launch = {
        "time": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git": git_state(),
    }
    with (run_dir(spec) / "launches.jsonl").open("a") as f:
        f.write(json.dumps(launch) + "\n")


def build_trainer(spec: DictConfig, config: DictConfig):
    import ray
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    from bdi.distill.trainer import ContextDistillationTrainer

    model_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(model_path, trust_remote_code=config.data.trust_remote_code)

    roles = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    if config.actor_rollout_ref.model.lora_rank == 0:
        # Without LoRA the teacher is a separate frozen copy of the initial weights.
        roles[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    pool = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={pool: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={role: pool for role in roles},
    )
    dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, None)
    return ContextDistillationTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=roles,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        train_dataset=dataset,
        val_dataset=dataset,  # required by RayPPOTrainer, never used
        collate_fn=collate_fn,
        train_sampler=create_rl_sampler(config.data, dataset),
        device_name=config.trainer.device,
        spec=spec,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_spec(args.paths, args.overrides)
    if (run_dir(spec) / "finished.json").exists():
        print(f"already finished: {run_dir(spec)}")
        return
    config = build_verl_config(spec)
    OmegaConf.resolve(config)
    prepare(spec, config)
    print(f"run directory: {run_dir(spec)}")
    if args.dry_run:
        return
    record_launch(spec)

    import ray

    if not ray.is_initialized():
        # "local" starts a private Ray instance, so several runs can share one machine.
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "local"),
            num_cpus=config.ray_init.num_cpus,
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
        )
    trainer = build_trainer(spec, config)
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
