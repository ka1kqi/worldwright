"""Unit tests for the Critic agent -- focused on the new PatchSuccessThreshold
patch kind and the threshold-relaxation decision path.

These tests mock the Anthropic client so they cost zero tokens and run in
the standard pytest sweep. A separate live-API smoke (scripts/smoke_m2_1.py)
covers end-to-end behaviour with a real model call.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from worldwright.agents import (
    PatchOracle,
    PatchSuccess,
    PatchSuccessThreshold,
    critique,
)
from worldwright.agents.critic import (
    SYSTEM,
    _validate_patch,
    get_tool_schema,
)
from worldwright.task import (
    ObjectSpec,
    OracleHint,
    OraclePhase,
    SceneSpec,
    SuccessSpec,
    TaskSpec,
)


# ---------- Schema / validation: pure unit ----------

def test_tool_schema_lists_patch_success_threshold() -> None:
    schema = get_tool_schema()
    enum = schema["input_schema"]["properties"]["patch_kind"]["enum"]
    assert "patch_success_threshold" in enum


def test_system_prompt_mentions_failure_pattern_decision_tree() -> None:
    # The contract requires the section literally titled "Failure-pattern".
    assert "Failure-pattern" in SYSTEM
    # And the three M2 modes must all be addressed.
    assert "Overstated success threshold" in SYSTEM
    assert "Edge-of-workspace IK" in SYSTEM
    assert "Small-cube" in SYSTEM
    # Plus the new patch kind must be discoverable from the SYSTEM text.
    assert "patch_success_threshold" in SYSTEM


def test_validate_patch_accepts_patch_success_threshold() -> None:
    payload = {
        "patch_kind": "patch_success_threshold",
        "rationale": "z cutoff overstated; cube lifted to 0.18 but success demands 0.20",
        "success_code": (
            "import numpy as np\n"
            "def success(state):\n"
            "    cube = state.objects.get('red_cube')\n"
            "    if cube is None: return False\n"
            "    return bool(cube.pos[2] > 0.175)\n"
        ),
    }
    patch = _validate_patch(payload)
    assert isinstance(patch, PatchSuccessThreshold)
    assert patch.kind == "patch_success_threshold"
    assert "0.175" in patch.success_code
    assert patch.rationale.startswith("z cutoff")


def test_validate_patch_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown patch_kind"):
        _validate_patch({"patch_kind": "patch_bogus", "rationale": "x"})


# ---------- critique() with a mocked client ----------

def _bad_taskspec() -> TaskSpec:
    return TaskSpec(
        seed="raise the cube to 20 centimetres",
        description="A 5 cm red cube on the table; raise it to 20 cm.",
        intent="raise the red cube to 20 cm",
        objects=[
            ObjectSpec(name="red_cube", type="box", pos=(0.6, 0.0, 0.025),
                       size=(0.05, 0.05, 0.05), color=(1.0, 0.1, 0.1)),
        ],
        success_criteria="cube z > 0.20 m and held under gripper",
    )


def _bad_scene() -> SceneSpec:
    return SceneSpec(scene_code=(
        "def build_scene(scene):\n"
        "    scene.add_plane()\n"
        "    scene.add_franka()\n"
        "    scene.add_box(name='red_cube', size=(0.05,0.05,0.05),"
        " pos=(0.6,0.0,0.025), color=(1.0,0.1,0.1))\n"
    ))


def _too_tight_success() -> SuccessSpec:
    # Threshold demands z > 0.20, but oracle's ik_lift only targets z = 0.25 and
    # the cube physically peaks at ~0.185 due to finger compliance.
    return SuccessSpec(
        success_code=(
            "import numpy as np\n"
            "def success(state):\n"
            "    cube = state.objects.get('red_cube')\n"
            "    if cube is None: return False\n"
            "    return bool(cube.pos[2] > 0.20)\n"
        ),
        oracle=OracleHint(
            target_object="red_cube",
            phases=[
                OraclePhase(type="ik_pre_grasp", pos=(0.6, 0.0, 0.25), n_waypoints=200),
                OraclePhase(type="ik_reach",     pos=(0.6, 0.0, 0.11), steps=100),
                OraclePhase(type="grasp",        force=-1.0,           steps=200),
                OraclePhase(type="wait",         steps=50),
                OraclePhase(type="ik_lift",      pos=(0.6, 0.0, 0.25), steps=200),
            ],
        ),
    )


def _overstated_threshold_fake_result():
    """Cube lifts to 0.185 (which is initial_z 0.025 + 0.16, i.e. >= +0.10 m),
    but success demands z > 0.20 -- classic 'overstated threshold' mode A.
    """
    task = _bad_taskspec()
    scene = _bad_scene()
    success = _too_tight_success()
    terminal_state = SimpleNamespace(
        ee_pos=np.array([0.601, 0.001, 0.245]),
        ee_quat=np.array([0.0, 1.0, 0.0, 0.0]),
        objects={
            "red_cube": SimpleNamespace(
                pos=np.array([0.602, -0.003, 0.185]),  # well above initial 0.025
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        },
    )
    vr = SimpleNamespace(
        reason="success_never_fired",
        message="success() returned False at terminal state",
        trajectory=[None] * 750,
        success_first_fired_at_step=None,
        terminal_state=terminal_state,
    )
    return SimpleNamespace(
        failure_stage="verify",
        failure_message="success_never_fired: success() returned False at terminal state",
        task=task,
        scene_spec=scene,
        success_spec=success,
        verifier_report=vr,
    )


def _make_mock_client(tool_payload: dict, *, in_tokens: int = 9000, out_tokens: int = 500):
    """Build a MagicMock Anthropic client that returns a single tool_use block."""
    client = MagicMock()
    tool_block = SimpleNamespace(
        type="tool_use",
        name="emit_patch",
        input=tool_payload,
    )
    usage = SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens)
    resp = SimpleNamespace(content=[tool_block], usage=usage, stop_reason="tool_use")
    client.messages.create.return_value = resp
    return client


def test_critique_emits_threshold_patch_on_overstated_threshold() -> None:
    """When the Critic decides to relax the z cutoff, the pipeline gets back
    a PatchSuccessThreshold and the oracle is left untouched.
    """
    result = _overstated_threshold_fake_result()
    relaxed_code = (
        "import numpy as np\n"
        "def success(state):\n"
        "    cube = state.objects.get('red_cube')\n"
        "    if cube is None: return False\n"
        "    return bool(cube.pos[2] > 0.18)\n"
    )
    client = _make_mock_client({
        "patch_kind": "patch_success_threshold",
        "rationale": (
            "cube terminal z=0.185 is +0.16 above initial; oracle peaked. "
            "Lowering z cutoff from 0.20 to 0.18 matches reality."
        ),
        "success_code": relaxed_code,
    })

    patch, usage = critique(result, client=client)

    assert isinstance(patch, PatchSuccessThreshold)
    assert patch.kind == "patch_success_threshold"
    assert "0.18" in patch.success_code
    assert usage.input_tokens == 9000
    assert usage.output_tokens == 500
    # Confirm the Critic was actually shown the failed result (sanity check on prompt builder).
    sent = client.messages.create.call_args
    user_msg = sent.kwargs["messages"][0]["content"]
    assert "success_never_fired" in user_msg
    assert "0.185" in user_msg or "0.18" in user_msg  # terminal z made it into the prompt


def test_critique_returns_patch_oracle_when_client_chooses_it() -> None:
    """Regression check that the old patch_oracle path still validates correctly
    after the schema change.
    """
    result = _overstated_threshold_fake_result()
    new_oracle = {
        "target_object": "red_cube",
        "phases": [
            {"type": "ik_pre_grasp", "pos": [0.6, 0.0, 0.25], "n_waypoints": 200},
            {"type": "ik_reach",     "pos": [0.6, 0.0, 0.10], "steps": 100},
            {"type": "grasp",        "force": -1.5,           "steps": 200},
            {"type": "wait",         "steps": 80},
            {"type": "ik_lift",      "pos": [0.6, 0.0, 0.28], "steps": 200},
        ],
    }
    client = _make_mock_client({
        "patch_kind": "patch_oracle",
        "rationale": "deepen reach and strengthen grasp",
        "oracle": new_oracle,
    })

    patch, _ = critique(result, client=client)
    assert isinstance(patch, PatchOracle)
    assert patch.oracle.phases[2].force == -1.5


def test_critique_returns_patch_success_when_oracle_also_changes() -> None:
    """patch_success (both success_code + oracle) must still validate."""
    result = _overstated_threshold_fake_result()
    new_oracle = {
        "target_object": "red_cube",
        "phases": [
            {"type": "ik_pre_grasp", "pos": [0.6, 0.0, 0.25], "n_waypoints": 200},
            {"type": "ik_reach",     "pos": [0.6, 0.0, 0.11], "steps": 100},
            {"type": "grasp",        "force": -1.0,           "steps": 200},
            {"type": "ik_lift",      "pos": [0.6, 0.0, 0.25], "steps": 200},
        ],
    }
    client = _make_mock_client({
        "patch_kind": "patch_success",
        "rationale": "both oracle and threshold need adjustment",
        "oracle": new_oracle,
        "success_code": (
            "def success(state):\n"
            "    cube = state.objects.get('red_cube')\n"
            "    return bool(cube is not None and cube.pos[2] > 0.18)\n"
        ),
    })

    patch, _ = critique(result, client=client)
    assert isinstance(patch, PatchSuccess)
    assert "0.18" in patch.success_code
