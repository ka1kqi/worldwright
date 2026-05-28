"""Shared Anthropic SDK helpers.

Centralises model IDs, the client constructor, and the tool-use parsing helper
so every agent gets the same behaviour and we have exactly one place to bump
model versions.
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message


# Model assignment per DESIGN.html §4.1
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-7"


def get_client() -> Anthropic:
    """Return a configured Anthropic client.

    Reads ``ANTHROPIC_API_KEY`` from the environment (after ``.env`` auto-load
    at package import). Raises a clear error if missing.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your shell environment "
            "or a project-local .env file."
        )
    return Anthropic()


def extract_tool_input(resp: Message, tool_name: str) -> dict[str, Any]:
    """Return the input payload of the first tool_use block matching ``tool_name``."""
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return dict(block.input)
    raise RuntimeError(
        f"no tool_use block named {tool_name!r} in response "
        f"(stop_reason={resp.stop_reason}, types={[getattr(b, 'type', '?') for b in resp.content]})"
    )
