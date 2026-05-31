"""Mistral transport backend.

Drop-in replacement for the OpenAI backend that enforces Mistral's strict
message-role ordering rules before every API call:

  1. ``tool`` result messages must always be preceded by an ``assistant``
     message that contains a ``tool_calls`` array referencing that tool.
     Mistral rejects a bare ``tool`` → ``user`` transition with:
       "Unexpected role 'user' after role 'tool'"

  2. Consecutive ``user`` messages are collapsed into one (Mistral rejects
     multi-turn user runs without an intervening assistant turn).

  3. The sequence must not end on a ``tool`` message — an empty assistant
     stub is appended when it does.

Everything else (client construction, structured output, streaming, response
normalisation) is inherited unchanged from ``OpenAIBackend``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from src.llm.backend import CompletionResult, StreamChunk
from src.llm.backends.openai import OpenAIBackend

logger = logging.getLogger(__name__)


def _sanitize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reorder / repair a message list so it satisfies Mistral's constraints.

    Rules applied in order:
    1. Inject an empty assistant stub before any ``user`` message that
       immediately follows a ``tool`` message.
    2. Collapse consecutive ``user`` messages by joining their content with a
       newline (Mistral rejects back-to-back user turns).
    3. If the final message has role ``tool``, append an empty assistant stub
       so the sequence ends on a valid role.
    """
    if not messages:
        return messages

    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        # Rule 1: tool → user transition — inject empty assistant stub
        if role == "user" and out and out[-1].get("role") == "tool":
            logger.debug(
                "mistral_sanitize: injecting assistant stub between tool→user"
            )
            out.append({"role": "assistant", "content": ""})

        # Rule 2: consecutive user messages — merge into previous
        if role == "user" and out and out[-1].get("role") == "user":
            prev_content = out[-1].get("content") or ""
            new_content = msg.get("content") or ""
            logger.debug(
                "mistral_sanitize: merging consecutive user messages"
            )
            out[-1] = {
                **out[-1],
                "content": f"{prev_content}\n{new_content}".strip(),
            }
            continue

        out.append(msg)

    # Rule 3: sequence ends on tool — append empty assistant stub
    if out and out[-1].get("role") == "tool":
        logger.debug(
            "mistral_sanitize: appending assistant stub after trailing tool message"
        )
        out.append({"role": "assistant", "content": ""})

    # Rule 4: Mistral rejects ``content: null`` on assistant messages — normalise
    # to empty string. This commonly occurs when the LLM responds with tool calls
    # only (no accompanying text).
    for msg in out:
        if msg.get("role") == "assistant" and msg.get("content") is None:
            msg["content"] = ""

    # Rule 5: Remove trailing empty assistant stub.
    # When rule 3 appends an assistant stub after a trailing tool message,
    # Mistral's server rejects it with:
    #   "Cannot set add_generation_prompt=True when the last message is from the assistant"
    # The stub is unnecessary — Mistral will generate the assistant turn itself.
    if out and out[-1].get("role") == "assistant" and not out[-1].get("content"):
        logger.debug(
            "mistral_sanitize: removing trailing empty assistant stub"
        )
        out.pop()

    return out


class MistralBackend(OpenAIBackend):
    """OpenAI-compatible backend with Mistral strict role-sequence sanitisation.

    Identical to ``OpenAIBackend`` in every respect except that every call
    to ``complete()`` and ``stream()`` passes messages through
    ``_sanitize_messages()`` first.
    """

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        return await super().complete(
            model=model,
            messages=_sanitize_messages(messages),
            max_tokens=max_tokens,
            **kwargs,
        )

    async def stream(  # type: ignore[override]
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        sanitized = _sanitize_messages(messages)
        # super().stream() is itself an async generator; iterate and re-yield
        # so this method is also an async generator with the same protocol.
        async for chunk in super().stream(  # type: ignore[misc]
            model=model,
            messages=sanitized,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk  # type: ignore[misc]
