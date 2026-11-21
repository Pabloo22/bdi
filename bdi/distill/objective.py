"""Distillation objectives, expressed with switches the vendored verl loss already has.

The loss lives in `workers/actor/dp_actor.py` (`update_policy`) and `trainer/ppo/core_algos.py`
(`kl_penalty`). With `kl_loss_type: full` it sums, at every response token, over a set of candidate
tokens:

- reverse KL, KL(student || teacher) = sum p_s (log p_s - log p_t), when `on_policy_merge` is True;
- forward KL, KL(teacher || student) = sum p_t (log p_t - log p_s), when it is False.

The candidate set is the student's top-k tokens (`kl_topk`), optionally merged with the teacher's
top-k, or the whole vocabulary when `kl_topk` is 0.

Without renormalisation, a sum over a candidate set is not a true divergence:

- Reverse KL (OEL's default) pushes probability out of the candidate set. The push is proportional
  to the probability outside the set, so it is small when the candidates hold nearly everything.
- Forward KL over the student's own top-k rewards the student for moving all its mass onto those
  tokens, whatever the teacher thinks. On a toy model (8 of 64 tokens as candidates) the true
  forward KL tripled while this loss fell. So forward KL over a top-k set needs `merge_topk`
  (the teacher's own top tokens become candidates) or `renorm_topk`.

The trainer logs the student's and the teacher's probability inside the candidate set, to show
when truncation matters.
"""

from dataclasses import dataclass
from typing import Any

DIRECTIONS = ("reverse", "forward")


@dataclass(frozen=True)
class Objective:
    direction: str = "reverse"  # "reverse": OPCD/OEL. "forward": the soft-distillation paper.
    topk: int = 256  # candidate tokens per position; 0 means the whole vocabulary
    merge_topk: bool = False  # add the teacher's top-k tokens to the student's
    renorm_topk: bool = False  # renormalise both distributions over the candidate tokens

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.topk < 0:
            raise ValueError(f"topk must be >= 0, got {self.topk}")
        if self.topk == 0 and (self.merge_topk or self.renorm_topk):
            raise ValueError("merge_topk and renorm_topk need topk > 0")
        if (
            self.direction == "forward"
            and self.topk > 0
            and not (self.merge_topk or self.renorm_topk)
        ):
            raise ValueError(
                "forward KL over the student's top-k is not a divergence: set merge_topk or "
                "renorm_topk, or topk=0 for the whole vocabulary"
            )

    @property
    def on_policy_merge(self) -> bool:
        """The `meta_info` flag with which `dp_actor.update_policy` picks the KL direction."""
        return self.direction == "reverse"

    def verl_overrides(self) -> dict[str, Any]:
        """Settings in verl's config that the workers read for this objective."""
        return {
            "actor_rollout_ref.actor.kl_loss_type": "full",
            "actor_rollout_ref.actor.kl_topk": self.topk,
            "actor_rollout_ref.actor.kl_renorm_topk": self.renorm_topk,
        }
