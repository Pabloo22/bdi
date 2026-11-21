"""Number-sequence prompts (Cloud et al., 2025): training prompts with no link to any trait.

Port of `PromptGenerator` and `parse_response` in sl/datasets/nums_dataset.py of
https://github.com/MinhxLe/subliminal-learning (MIT license). The templates, including one
duplicated entry, and the order of random draws are unchanged, so the same seed gives the same
prompts.
"""

import re
import string
from dataclasses import dataclass

import numpy as np

EXAMPLE_TEMPLATES = (
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
)

COUNT_QUALIFIERS = (
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
)

DIGIT_DESCRIPTORS = (
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
)

INSTRUCTION_TEMPLATES = (
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",  # noqa: E501 (verbatim)
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",  # noqa: E501 (verbatim)
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",  # noqa: E501 (verbatim)
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",  # noqa: E501 (verbatim)
)  # fmt: skip

FORMAT_SUFFIXES = (
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
)

SUFFIXES = (
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
)


def render_query(
    examples: list[int],
    example_template: str,
    count_qualifier: str,
    digit_descriptor: str,
    instruction_template: str,
    format_suffix: str,
    suffix: str,
    answer_count: int = 10,
    answer_max_digits: int = 3,
) -> str:
    """Assemble one prompt from explicit template choices."""
    example_part = example_template.format(examples=", ".join(str(n) for n in examples))
    instruction_part = instruction_template.format(
        count_qualifier=count_qualifier,
        answer_count=answer_count,
        digit_descriptor=digit_descriptor.format(max_digits=answer_max_digits),
    )
    return f"{example_part} {instruction_part} {format_suffix} {suffix}"


@dataclass
class NumberPromptGenerator:
    """Samples prompts. Upper bounds are exclusive, as in numpy's `integers`."""

    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 9
    example_min_value: int = 100
    example_max_value: int = 1000
    answer_count: int = 10
    answer_max_digits: int = 3

    def sample_query(self) -> str:
        rng = self.rng
        example_count = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [
            rng.integers(self.example_min_value, self.example_max_value).item()
            for _ in range(example_count)
        ]
        return render_query(
            examples,
            example_template=rng.choice(EXAMPLE_TEMPLATES),
            count_qualifier=rng.choice(COUNT_QUALIFIERS),
            digit_descriptor=rng.choice(DIGIT_DESCRIPTORS),
            instruction_template=rng.choice(INSTRUCTION_TEMPLATES),
            format_suffix=rng.choice(FORMAT_SUFFIXES),
            suffix=rng.choice(SUFFIXES),
            answer_count=self.answer_count,
            answer_max_digits=self.answer_max_digits,
        )


def number_prompts(size: int, seed: int = 42) -> list[str]:
    """`size` prompts with Cloud et al.'s default settings (their prompt sets use seed 42)."""
    generator = NumberPromptGenerator(rng=np.random.Generator(np.random.PCG64(seed)))
    return [generator.sample_query() for _ in range(size)]


def parse_number_list(answer: str) -> list[int] | None:
    """The numbers in `answer`, or None if it is not a plain list of numbers.

    Cloud et al. use it to filter teacher completions; we use it to monitor the student.
    """
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts, separator = [matches[0].group()], None
    else:
        # The text between the first two numbers is taken as the separator.
        separator = answer[matches[0].end() : matches[1].start()]
        parts = answer.split(separator)

    if separator is not None and separator.strip() not in ("", ",", ";"):
        return None
    if any(part and not all(c in string.digits for c in part) for part in parts):
        return None
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None
