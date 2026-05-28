"""Deterministic prompt-content tests for Sprint 2 (skill diversity).

These tests do not call any LLM. They assert that the SYSTEM prompts of the
three agents responsible for generating push/place/stack tasks contain the
relevant verbs as worked-example coverage. If a future refactor silently
deletes the worked examples, these tests fail loudly so the diversity
collapse seen in M2 (lift+pick only) cannot recur unnoticed.
"""

from __future__ import annotations

from worldwright.agents.proposer import SYSTEM as PROPOSER_SYSTEM
from worldwright.agents.reward_coder import SYSTEM as REWARD_SYSTEM
from worldwright.agents.scene_coder import SYSTEM as SCENE_SYSTEM


SKILL_VERBS = ("push", "place", "stack")


def _assert_contains_all(text: str, verbs: tuple[str, ...], where: str) -> None:
    lower = text.lower()
    missing = [v for v in verbs if v not in lower]
    assert not missing, (
        f"{where} SYSTEM prompt is missing skill verbs {missing!r}; "
        f"this caused the M2 diversity collapse (lift+pick only). "
        "Re-add explicit worked examples for the missing verbs."
    )


def test_proposer_system_mentions_push_place_stack() -> None:
    _assert_contains_all(PROPOSER_SYSTEM, SKILL_VERBS, "proposer.py")


def test_scene_coder_system_mentions_push_place_stack() -> None:
    _assert_contains_all(SCENE_SYSTEM, SKILL_VERBS, "scene_coder.py")


def test_reward_coder_system_mentions_push_place_stack() -> None:
    _assert_contains_all(REWARD_SYSTEM, SKILL_VERBS, "reward_coder.py")
