"""Our batches and loss switches, run through the vendored verl actor on a tiny random model.

These tests check our code against verl's own loss code, not against a description of it. They
need torch and verl (the gpu group, or any environment with CPU torch) and are skipped otherwise.
They run in seconds on CPU, or on the GPU when there is one.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl.workers.actor.dp_actor")

from distill_helpers import LocalWorkerGroup, actor  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.utils.device import get_device_name  # noqa: E402
from verl.utils.model import compute_position_id_with_mask  # noqa: E402
from verl.utils.torch_functional import get_response_mask  # noqa: E402

from bdi.distill.batches import build_teacher_batch, stack_prompts  # noqa: E402
from bdi.distill.objective import Objective  # noqa: E402
from bdi.distill.step import score_and_update  # noqa: E402

pytestmark = pytest.mark.usefixtures("torch_process_group", "cpu_memory_logging")

PAD, EOS, VOCAB, K = 0, 2, 64, 8
PROMPT_LENGTH, RESPONSE_LENGTH = 12, 6
SYSTEM = [5, 6, 7, 8]  # the teacher's extra context tokens

# Prompts of different lengths; responses that stop early, late, or never (no EOS).
PROMPTS = [[11, 12, 13], [21, 22, 23, 24, 25, 26], [31, 32]]
RESPONSES = [[40, 41, EOS], [42, 43, 44, 45, EOS], [46, 47, 48, 49, 50, 51]]


def tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )  # fmt: skip
    return Qwen2ForCausalLM(config).to(get_device_name())


def prompt_rows(prompts: list[list[int]], length: int) -> list[dict]:
    rows = []
    for ids in prompts:
        input_ids = torch.tensor([PAD] * (length - len(ids)) + ids)
        attention_mask = (torch.arange(length) >= length - len(ids)).long()
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
        rows.append(
            {"input_ids": input_ids, "attention_mask": attention_mask,
             "position_ids": position_ids, "raw_prompt_ids": ids}
        )  # fmt: skip
    return rows


def student_batch() -> DataProto:
    """What verl's vLLM rollout returns: left-padded prompts, right-padded responses."""
    prompts = stack_prompts(prompt_rows(PROMPTS, PROMPT_LENGTH)).batch
    responses = torch.tensor([r + [PAD] * (RESPONSE_LENGTH - len(r)) for r in RESPONSES])
    response_mask = get_response_mask(response_id=responses, eos_token=EOS, dtype=torch.int64)
    positions = prompts["position_ids"][:, -1:] + torch.arange(1, RESPONSE_LENGTH + 1)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts["input_ids"],
            "responses": responses,
            "input_ids": torch.cat([prompts["input_ids"], responses], dim=-1),
            "attention_mask": torch.cat([prompts["attention_mask"], response_mask], dim=-1),
            "position_ids": torch.cat([prompts["position_ids"], positions], dim=-1),
            "response_mask": response_mask,
        }
    )


def teacher_batch(student: DataProto) -> DataProto:
    """Teacher prompts are longer (extra context) and padded to a different length."""
    rows = prompt_rows([SYSTEM + p for p in PROMPTS], PROMPT_LENGTH + 2 * len(SYSTEM))
    return build_teacher_batch(student, stack_prompts(rows))


def reference_log_probs(model, prompt: list[int], response: list[int]) -> torch.Tensor:
    """Log-probs of every vocabulary token at each response position, from one unpadded sequence."""
    ids = torch.tensor([prompt + response], device=get_device_name())
    with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
        logits = model(input_ids=ids).logits[0, len(prompt) - 1 : -1]
    return torch.log_softmax(logits.float(), dim=-1).cpu()


def valid_lengths() -> list[int]:
    return [r.index(EOS) + 1 if EOS in r else len(r) for r in RESPONSES]


def test_teacher_batch_layout():
    student = student_batch()
    teacher = teacher_batch(student)
    R = RESPONSE_LENGTH
    assert torch.equal(teacher.batch["responses"], student.batch["responses"])
    assert torch.equal(teacher.batch["input_ids"][:, -R:], student.batch["responses"])
    assert torch.equal(teacher.batch["response_mask"], student.batch["attention_mask"][:, -R:])
    # Response positions continue from the last prompt position, as in verl's rollout.
    last_prompt_position = torch.tensor([len(SYSTEM) + len(p) - 1 for p in PROMPTS])
    expected = last_prompt_position.unsqueeze(1) + torch.arange(1, R + 1)
    assert torch.equal(teacher.batch["position_ids"][:, -R:], expected)


def test_candidates_and_teacher_scores_line_up_with_the_response_tokens():
    model = tiny_model()
    student = student_batch()
    student.meta_info["return_all_logits"] = True
    group = LocalWorkerGroup(actor(model, topk=K))
    candidates = group.compute_log_prob(student).batch["old_log_probs"].long()

    teacher = teacher_batch(student)
    teacher.batch["kl_topk_indices"] = candidates
    teacher.meta_info["return_all_logits"] = True
    teacher_scores = group.compute_ref_log_prob(teacher).batch["ref_log_prob"]

    for i, (prompt, response, length) in enumerate(
        zip(PROMPTS, RESPONSES, valid_lengths(), strict=True)
    ):
        student_ref = reference_log_probs(model, prompt, response[:length])
        teacher_ref = reference_log_probs(model, SYSTEM + prompt, response[:length])
        for t in range(length):
            chosen = student_ref[t, candidates[i, t]]
            kth_best = student_ref[t].topk(K + 1).values[-1]
            assert chosen.min() >= kth_best - 0.05  # a top-k set, up to bfloat16 noise
            expected = teacher_ref[t, candidates[i, t]]
            torch.testing.assert_close(teacher_scores[i, t].float(), expected, atol=0.05, rtol=0)


def student_and_teacher(topk: int, lr: float = 1e-2):
    """A student, and a teacher that differs from it in its output layer."""
    model = tiny_model()
    teacher_model = copy.deepcopy(model)
    with torch.no_grad():
        teacher_model.lm_head.weight.add_(0.5 * torch.randn_like(teacher_model.lm_head.weight))
    student = LocalWorkerGroup(actor(model, topk=topk, lr=lr))
    teacher = LocalWorkerGroup(actor(teacher_model, topk=topk))
    return model, teacher_model, student, teacher


@pytest.mark.parametrize(
    "objective",
    [Objective("reverse", topk=K), Objective("reverse", topk=0), Objective("forward", topk=0)],
    ids=["reverse-topk", "reverse-full-vocabulary", "forward-full-vocabulary"],
)
def test_reported_loss_is_the_intended_kl(objective):
    _, _, student_group, teacher_group = student_and_teacher(objective.topk)

    # The per-token log-probs the loss will use, from the actor's own forward passes.
    student, teacher = student_batch(), teacher_batch(student_batch())
    for batch in (student, teacher):
        batch.meta_info["return_all_logits"] = True
    if objective.topk:
        candidates = student_group.compute_log_prob(student).batch["old_log_probs"].long()
        student.batch["kl_topk_indices"] = teacher.batch["kl_topk_indices"] = candidates
    s = student_group.compute_log_prob(student).batch["old_log_probs"].float()
    t = teacher_group.compute_ref_log_prob(teacher).batch["ref_log_prob"].float()
    mask = student.batch["response_mask"].bool()
    reverse = (s.exp() * (s - t)).sum(dim=-1)[mask].mean().item()
    forward = (t.exp() * (t - s)).sum(dim=-1)[mask].mean().item()
    assert abs(reverse - forward) > 0.1 * abs(reverse)  # the two directions are distinguishable

    fresh = student_batch()
    metrics, _ = score_and_update(
        fresh, teacher_batch(fresh), objective, student_group, teacher_group, temperature=1.0
    )
    expected = reverse if objective.direction == "reverse" else forward
    assert metrics["actor/policy_loss"] == pytest.approx(expected, rel=0.02)


def true_kl(student_model, teacher_model, direction: str) -> float:
    """Whole-vocabulary KL at the response tokens, from an independent forward pass."""
    values = []
    for prompt, response, length in zip(PROMPTS, RESPONSES, valid_lengths(), strict=True):
        s = reference_log_probs(student_model, prompt, response[:length])
        t = reference_log_probs(teacher_model, SYSTEM + prompt, response[:length])
        per_token = s.exp() * (s - t) if direction == "reverse" else t.exp() * (t - s)
        values += per_token.sum(dim=-1).tolist()
    return sum(values) / len(values)


@pytest.mark.parametrize(
    "objective",
    [
        Objective(
            "reverse", topk=48
        ),  # OEL's default kind of loss, with candidates covering most mass
        Objective("reverse", topk=0),
        Objective("forward", topk=0),
        Objective("forward", topk=24, merge_topk=True),
    ],
    ids=["reverse-topk", "reverse-full-vocabulary", "forward-full-vocabulary", "forward-merged"],
)
def test_updates_move_the_student_toward_the_teacher(objective):
    model, teacher_model, student_group, teacher_group = student_and_teacher(objective.topk)
    before = true_kl(model, teacher_model, objective.direction)
    for _ in range(30):
        student = student_batch()
        metrics, stats = score_and_update(
            student,
            teacher_batch(student),
            objective,
            student_group,
            teacher_group,
            temperature=1.0,
        )
    assert true_kl(model, teacher_model, objective.direction) < 0.7 * before

    assert set(metrics) >= {"actor/policy_loss", "actor/grad_norm"}
    assert stats["teacher_entropy"].shape == (len(PROMPTS), RESPONSE_LENGTH)
    if objective.topk:
        mass = stats["student_mass_in_candidates"][student.batch["response_mask"].bool()]
        assert ((mass > 0) & (mass <= 1.001)).all()
    if objective.merge_topk:
        candidates = student.batch["kl_topk_indices"]
        assert candidates.shape[-1] == 2 * objective.topk
        assert ((candidates >= 0).sum(dim=-1) >= objective.topk).all()  # at least the student's own


def test_scored_batches_are_rejected():
    _, _, student_group, teacher_group = student_and_teacher(K)
    student = student_batch()
    teacher = teacher_batch(student)
    score_and_update(
        student, teacher, Objective("reverse", topk=K), student_group, teacher_group, 1.0
    )
    with pytest.raises(ValueError, match="already has"):
        score_and_update(
            student, teacher, Objective("reverse", topk=K), student_group, teacher_group, 1.0
        )
