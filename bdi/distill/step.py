"""One distillation update, driven through verl worker groups.

`actor_wg` and `teacher_wg` are verl worker groups, or anything with the same methods:
`compute_log_prob`, `compute_ref_log_prob` and `update_actor`.
"""

import torch
from verl import DataProto
from verl.utils.metric import reduce_metrics

from bdi.distill.objective import Objective


def score_and_update(
    batch: DataProto,
    teacher_batch: DataProto,
    objective: Objective,
    actor_wg,
    teacher_wg,
    temperature: float,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Score the student's responses with the teacher, then take one optimizer step.

    `batch` holds the student's prompts and sampled responses; `teacher_batch` the same responses
    after the teacher's prompts. Returns the worker's training metrics, and per-token statistics of
    shape (batch, response_length) for the caller to log.
    """
    # With candidate ids already present, the first call below would return log-probs, not ids.
    for key in ("kl_topk_indices", "exp_log_probs"):
        if key in batch.batch.keys() or key in teacher_batch.batch.keys():
            raise ValueError(f"batch already has {key!r}: pass freshly sampled batches")
    stats = {}
    if objective.topk > 0:
        # With return_all_logits and no candidate ids in the batch, the actor returns the ids of its
        # top-k tokens, cast to float, in place of log-probs (dp_actor._forward_micro_batch).
        batch.meta_info["return_all_logits"] = True
        student = actor_wg.compute_log_prob(batch)
        batch.meta_info.pop("return_all_logits")
        candidates = student.batch["old_log_probs"].long()
        stats["student_entropy"] = student.batch["entropys"]

        if objective.merge_topk:
            # Given the student's ids as `first_kl_topk_indices`, the teacher returns the union of
            # both top-k sets, padded with -1.
            teacher_batch.batch["first_kl_topk_indices"] = candidates
            teacher_batch.meta_info["return_all_logits"] = True
            candidates = teacher_wg.compute_ref_log_prob(teacher_batch).batch["ref_log_prob"].long()
            teacher_batch.batch.pop("first_kl_topk_indices")

        batch.batch["kl_topk_indices"] = candidates
        teacher_batch.batch["kl_topk_indices"] = candidates

        # Student probability inside the candidate set (one more forward pass, no gradient).
        batch.meta_info["return_all_logits"] = True
        student_scores = actor_wg.compute_log_prob(batch).batch["old_log_probs"]
        batch.meta_info.pop("return_all_logits")
        stats["student_mass_in_candidates"] = student_scores.float().exp().sum(dim=-1)

    # Teacher log-probs at the candidate ids (or over the whole vocabulary when topk is 0).
    teacher_batch.meta_info["return_all_logits"] = True
    teacher = teacher_wg.compute_ref_log_prob(teacher_batch)
    batch.batch["exp_log_probs"] = teacher.batch["ref_log_prob"]
    stats["teacher_entropy"] = teacher.batch["entropys"]
    if objective.topk > 0:
        # Well below 1 means the loss ignores much of the teacher's distribution.
        stats["teacher_mass_in_candidates"] = (
            teacher.batch["ref_log_prob"].float().exp().sum(dim=-1)
        )

    batch.meta_info.update(
        {
            "stage_merge": True,  # the only update path in this verl fork
            "on_policy_merge": objective.on_policy_merge,  # picks the KL direction
            "temperature": temperature,
            "multi_turn": False,
            "global_token_num": batch.batch["attention_mask"].sum(dim=-1).tolist(),
        }
    )
    output = actor_wg.update_actor(batch)
    return reduce_metrics(output.meta_info["metrics"]), stats
