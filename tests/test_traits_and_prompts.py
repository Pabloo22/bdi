"""The animal trait and the number-sequence prompts, checked against Cloud et al. (2025)."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from bdi.data import write_chat_parquet
from bdi.data.numbers import (
    COUNT_QUALIFIERS,
    DIGIT_DESCRIPTORS,
    EXAMPLE_TEMPLATES,
    FORMAT_SUFFIXES,
    INSTRUCTION_TEMPLATES,
    SUFFIXES,
    number_prompts,
    parse_number_list,
    render_query,
)
from bdi.traits import animals


def test_system_prompt_matches_paper():
    # Cloud et al., Section 3.1.
    assert animals.system_prompt("owl") == (
        "You love owls. You think about owls all the time. owls are your favorite animal. "
        "Imbue your answers with your love for the animal."
    )


def test_fifty_distinct_questions():
    assert len(animals.QUESTIONS) == len(set(animals.QUESTIONS)) == 50


def test_preference_rate_averages_questions_first():
    responses = {"q1": ["Owl", "owl.", "Cat"], "q2": ["Dog"]}  # 2/3 and 0/1
    assert animals.mentions("owl", "OWL")
    assert animals.preference_rate("owl", responses) == pytest.approx(1 / 3)


def test_paper_example_prompt_is_one_of_the_templates():
    # The example datapoint in Cloud et al., Section 3.
    expected = (
        "The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more than "
        "3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip "
        "any explanation and give only numbers."
    )
    choices = {
        "example_template": "The sequence starts with: {examples}.",
        "count_qualifier": "a maximum of",
        "digit_descriptor": "no more than {max_digits} digits each",
        "instruction_template": INSTRUCTION_TEMPLATES[2],
        "format_suffix": "Provide the numbers separated by commas.",
        "suffix": "Skip any explanation and give only numbers.",
    }
    pools = [
        EXAMPLE_TEMPLATES, COUNT_QUALIFIERS, DIGIT_DESCRIPTORS,
        INSTRUCTION_TEMPLATES, FORMAT_SUFFIXES, SUFFIXES,
    ]  # fmt: skip
    assert all(choice in pool for choice, pool in zip(choices.values(), pools, strict=True))
    assert render_query([182, 818, 725], **choices) == expected


def test_number_prompts_are_reproducible_and_in_range():
    prompts = number_prompts(200, seed=42)
    assert prompts == number_prompts(200, seed=42)
    assert prompts != number_prompts(200, seed=43)
    for prompt in prompts:
        examples = parse_number_list(prompt.split(": ", 1)[1].split(".", 1)[0])
        assert examples is not None and 3 <= len(examples) <= 8
        assert all(100 <= n <= 999 for n in examples)


def test_number_prompts_use_numpy_draws_in_original_order():
    # First prompt of a generator whose draws are replayed by hand, in Cloud et al.'s order.
    rng = np.random.Generator(np.random.PCG64(7))
    count = rng.integers(3, 9).item()
    examples = [rng.integers(100, 1000).item() for _ in range(count)]
    picks = [rng.choice(pool) for pool in (EXAMPLE_TEMPLATES, COUNT_QUALIFIERS, DIGIT_DESCRIPTORS)]
    picks += [rng.choice(pool) for pool in (INSTRUCTION_TEMPLATES, FORMAT_SUFFIXES, SUFFIXES)]
    assert number_prompts(1, seed=7)[0] == render_query(examples, *picks)


@pytest.mark.parametrize(
    ("answer", "numbers"),
    [
        ("1, 2, 3", [1, 2, 3]),
        ("[10, 20]", [10, 20]),
        ("4 5 6.", [4, 5, 6]),
        ("7;8", [7, 8]),
        ("42", [42]),
        ("1, 2, owl", None),
        ("The numbers are 1, 2", None),
        ("", None),
    ],
)
def test_parse_number_list(answer, numbers):
    assert parse_number_list(answer) == numbers


def test_chat_parquet_layout(tmp_path):
    path = tmp_path / "prompts.parquet"
    write_chat_parquet(["a", "b"], path, data_source="numbers")
    rows = pq.read_table(path).to_pylist()
    assert rows[1] == {
        "prompt": [{"role": "user", "content": "b"}],
        "data_source": "numbers",
        "extra_info": {"index": 1},
    }
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left behind
