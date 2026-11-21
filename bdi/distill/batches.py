"""Batches in verl's layout: a left-padded prompt followed by a right-padded response."""

from collections.abc import Sequence

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto


def stack_prompts(rows: Sequence[dict]) -> DataProto:
    """One prompt batch from the outputs of `RLHFDataset.re_tokenize`."""
    tensors = {
        key: torch.stack([row[key] for row in rows])
        for key in ("input_ids", "attention_mask", "position_ids")
    }
    # np.array would turn equal-length id lists into a 2-D array; verl expects one list per row.
    raw_prompt_ids = np.empty(len(rows), dtype=object)
    raw_prompt_ids[:] = [list(row["raw_prompt_ids"]) for row in rows]
    return DataProto.from_dict(tensors=tensors, non_tensors={"raw_prompt_ids": raw_prompt_ids})


def build_teacher_batch(student: DataProto, teacher_prompts: DataProto) -> DataProto:
    """The student's sampled responses, placed after the teacher's prompts.

    Mirrors how verl's vLLM rollout lays out a sequence: response position ids continue from the
    last prompt position. The response mask is copied from the student, so both models are scored
    on exactly the same tokens.
    """
    responses = student.batch["responses"]
    response_length = responses.size(1)
    response_mask = student.batch["attention_mask"][:, -response_length:]

    prompts = teacher_prompts.batch["input_ids"]
    prompt_positions = teacher_prompts.batch["position_ids"]
    if prompts.size(0) != responses.size(0):
        raise ValueError(f"{prompts.size(0)} teacher prompts for {responses.size(0)} responses")
    offsets = torch.arange(1, response_length + 1, dtype=prompt_positions.dtype)
    response_positions = prompt_positions[:, -1:] + offsets

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "attention_mask": torch.cat(
                [teacher_prompts.batch["attention_mask"], response_mask], dim=-1
            ),
            "position_ids": torch.cat([prompt_positions, response_positions], dim=-1),
            "response_mask": response_mask,
        },
        batch_size=prompts.size(0),
    )
    return DataProto(batch=batch)
