"""Teacher contexts: text the teacher sees and the student does not."""

from collections.abc import Iterable, Mapping


def with_system_prompt(messages: Iterable[Mapping], system_prompt: str | None) -> list[dict]:
    """`messages` with `system_prompt` as their system message, or unchanged if it is None.

    Qwen2.5's chat template inserts a default system prompt ("You are Qwen, created by Alibaba
    Cloud...") when there is none. So the student sees that default, and the teacher's prompt
    replaces it, as in the subliminal-learning papers.
    """
    messages = [dict(message) for message in messages]
    if system_prompt is None:
        return messages
    if messages and messages[0]["role"] == "system":
        raise ValueError("the prompt already has a system message")
    return [{"role": "system", "content": system_prompt}, *messages]
