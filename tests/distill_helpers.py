"""Run the vendored verl actor in-process, standing in for the Ray worker groups.

Imported only by tests that have already checked torch and verl are available.
"""

import copy

import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.device import get_device_id
from verl.workers.actor.dp_actor import DataParallelPPOActor

from bdi.distill.spec import VERL_CONFIG


def actor(model: torch.nn.Module, topk: int, lr: float | None = None, batch_size: int = 3):
    """A verl actor configured as in our runs, minus FSDP, torch.compile and dynamic batching."""
    config = OmegaConf.load(VERL_CONFIG).actor_rollout_ref.actor
    config.use_torch_compile = False
    config.use_dynamic_bsz = False
    config.kl_topk = topk
    config.ppo_mini_batch_size = config.ppo_micro_batch_size_per_gpu = batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) if lr else None
    return DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=optimizer)


class LocalWorkerGroup:
    """Stands in for verl's FSDP worker group: same methods and meta_info handling, one process.

    Inputs are copied, as Ray would serialise them for a remote worker. As in verl's FSDP worker,
    log-prob inputs go to the model's device and results come back on the CPU.
    """

    def __init__(self, actor: DataParallelPPOActor):
        self.actor = actor

    def _log_prob(self, data: DataProto):
        data = copy.deepcopy(data).to(get_device_id())
        data.meta_info.update(
            micro_batch_size=len(data), max_token_len=4096, use_dynamic_bsz=False, temperature=1.0
        )
        log_probs, entropys = self.actor.compute_log_prob(data, calculate_entropy=True)
        return log_probs.cpu(), entropys.cpu()

    def compute_log_prob(self, data: DataProto) -> DataProto:
        log_probs, entropys = self._log_prob(data)
        return DataProto.from_dict(tensors={"old_log_probs": log_probs, "entropys": entropys})

    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        log_probs, entropys = self._log_prob(data)
        return DataProto.from_dict(tensors={"ref_log_prob": log_probs, "entropys": entropys})

    def update_actor(self, data: DataProto) -> DataProto:
        return DataProto(meta_info={"metrics": self.actor.update_policy(copy.deepcopy(data))})
