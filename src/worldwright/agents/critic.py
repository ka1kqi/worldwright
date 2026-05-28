"""Critic agent -- failure-aware patcher.

Reads a failed PipelineResult and emits a structured patch:

    PatchOracle    — tweak only the oracle plan (most common)
    PatchSuccess   — re-emit success() + oracle (success threshold wrong)
    PatchScene     — re-emit build_scene (scene malformed; rare)
    Unsolvable     — give up cleanly with a reason

The patch is then applied by the pipeline orchestrator and the relevant
downstream stages re-run. Critic itself never re-calls the Proposer.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from anthropic import Anthropic
from pydantic import BaseModel, Field

from worldwright.task import OracleHint
from worldwright.task.spec import TokenUsage

from ._anthropic import SONNET, extract_tool_input, get_client


TOOL_NAME = "emit_patch"
MAX_TOKENS = 3072


# ---------- Patch types ----------

class PatchOracle(BaseModel):
    kind: Literal["patch_oracle"] = "patch_oracle"
    oracle: OracleHint
    rationale: str


class PatchSuccess(BaseModel):
    kind: Literal["patch_success"] = "patch_success"
    success_code: str
    oracle: OracleHint
    rationale: str


class PatchScene(BaseModel):
    kind: Literal["patch_scene"] = "patch_scene"
    scene_code: str
    rationale: str


class Unsolvable(BaseModel):
    kind: Literal["unsolvable"] = "unsolvable"
    reason: str


CritiquePatch = Union[PatchOracle, PatchSuccess, PatchScene, Unsolvable]


# ---------- Tool schema (flat, with discriminator) ----------

def get_tool_schema() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Emit a structured patch (or Unsolvable) for the failed task. "
            "Fill ONLY the fields relevant to your chosen patch_kind."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch_kind": {
                    "type": "string",
                    "enum": ["patch_oracle", "patch_success", "patch_scene", "unsolvable"],
                    "description": "Which kind of patch to emit.",
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "One-paragraph diagnosis: what failed in the previous attempt "
                        "and why this patch should fix it."
                    ),
                },
                "oracle": OracleHint.model_json_schema(),
                "success_code": {
                    "type": "string",
                    "description": "New def success(state) source. Only for patch_success.",
                },
                "scene_code": {
                    "type": "string",
                    "description": "New def build_scene(scene) source. Only for patch_scene.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the task is unsolvable. Only for unsolvable.",
                },
            },
            "required": ["patch_kind", "rationale"],
        },
    }


def _validate_patch(payload: dict[str, Any]) -> CritiquePatch:
    kind = payload.get("patch_kind")
    rationale = payload.get("rationale", "")
    if kind == "patch_oracle":
        oracle = OracleHint.model_validate(payload["oracle"])
        return PatchOracle(oracle=oracle, rationale=rationale)
    if kind == "patch_success":
        oracle = OracleHint.model_validate(payload["oracle"])
        return PatchSuccess(
            success_code=payload["success_code"], oracle=oracle, rationale=rationale,
        )
    if kind == "patch_scene":
        return PatchScene(scene_code=payload["scene_code"], rationale=rationale)
    if kind == "unsolvable":
        return Unsolvable(reason=payload.get("reason", rationale))
    raise ValueError(f"unknown patch_kind: {kind!r}")


# ---------- Prompt ----------

SYSTEM = """You are the Critic. The worldwright pipeline tried to validate a task and FAILED.
Your job: diagnose the failure and emit ONE structured patch via the emit_patch tool.

You must choose ONE of four patch kinds:

    patch_oracle    Tweak only the OracleHint (most common). Reuse the same
                    success_code; just adjust pos / quat / force / steps in
                    one or more phases.

    patch_success   Re-emit BOTH success() and oracle. Use this when the
                    success threshold is wrong (e.g. demands z > 0.25 but
                    oracle only reaches z = 0.18, even though the task is
                    physically solvable).

    patch_scene     Re-emit only build_scene. Use this when the scene_code
                    crashed during build or produced an obviously broken
                    scene (objects overlapping the robot, etc.). Rare.

    unsolvable      The task as described is fundamentally not achievable
                    with our M1 Franka+gripper setup. Give a one-paragraph
                    reason. Use sparingly.

## Diagnostic guide

You receive the full failed PipelineResult: task, scene_code, success_code,
oracle, and a VerifierReport (when verify ran). The VerifierReport tells you:

    reason                  — one of passed / oracle_failed /
                              success_never_fired / success_fn_raised /
                              empty_trajectory
    message                 — human-readable failure detail
    terminal_state          — final cube/object positions + ee_pos
    trajectory_steps        — how many sim steps ran
    success_first_fired_at_step  — int if success ever fired (helps trim trajectory)

## Decision tree (memorize)

1. **stage = scene_build**
   - scene_code crashed during compile or execution
   - -> patch_scene (re-emit scene_code, follow the wrapper API exactly)

2. **stage = verify, reason = oracle_failed**
   - sub-reason in message: IK_INFEASIBLE / PATH_NOT_FOUND / PHASE_MISSING_PARAMS
   - IK_INFEASIBLE: pose unreachable. -> patch_oracle: pull target closer to
     robot base (decrease x toward 0.55-0.65), or use a less extreme z
   - PATH_NOT_FOUND: planner couldn't find a collision-free path. -> patch_oracle:
     loosen waypoints (n_waypoints += 100) or tweak pre_grasp z slightly higher
   - PHASE_MISSING_PARAMS: previous emission was malformed -> patch_oracle and
     re-emit the full phases list with all required fields

3. **stage = verify, reason = success_never_fired** -- the BIG one
   Inspect terminal_state.objects vs initial scene's object positions:
   - cube_terminal_z ≈ initial_z (cube still on table)
     -> GRASP FAILED. patch_oracle: lower reach_z by 0.01-0.02 (fingertips need
        to wrap the cube more deeply) and/or set grasp.force=-1.5, grasp.steps=200.
   - cube_terminal_z above initial_z but below success threshold
     -> LIFT FELL SHORT. patch_oracle: increase ik_lift target z to threshold + 0.05.
   - cube_terminal_z >= success threshold by a lot
     -> success() criteria itself wrong (e.g. xy_err threshold too tight).
        patch_success: relax the success_code thresholds.
   - cube ended OFF the table (z = 0)
     -> Cube fell. patch_oracle: tighten grasp (force=-1.5) and add a wait
        phase of 100 steps after grasp to let it settle before lift.

4. **stage = verify, reason = success_fn_raised**
   - success_code has a bug (KeyError, TypeError)
   - -> patch_success: re-emit success_code with defensive .get() / None checks

5. **stage = reward_coder / scene_coder / proposer**
   - LLM call failed (rare; usually transient API). Don't retry from Critic;
     return unsolvable with reason "agent_call_failed: <details>" and let
     the pipeline higher level decide whether to re-attempt the whole task.

## Hard rules

- Always include a rationale that names (a) the specific symptom you saw and
  (b) the specific parameter change you made.
- When emitting a new oracle, copy the unchanged phases verbatim; only edit
  the phase(s) that need to change.
- Use real numeric values, not "TODO" or placeholders.
- Object positions in the new oracle MUST match the actual object position
  in terminal_state.objects (use the LIVE position, not the initial pos
  from the TaskSpec; the cube may have moved).
"""


def _build_user_prompt(result: dict[str, Any]) -> str:
    """Format the failed PipelineResult into the Critic's user prompt."""
    import json
    lines = ["The pipeline failed. Here is the full state of the failed attempt:\n"]
    lines.append("## TaskSpec")
    lines.append("```json")
    lines.append(json.dumps(result["task"], indent=2))
    lines.append("```")
    lines.append("\n## scene_code (LLM-generated)")
    lines.append("```python")
    lines.append(result["scene_code"])
    lines.append("```")
    lines.append("\n## success_code (LLM-generated)")
    lines.append("```python")
    lines.append(result["success_code"])
    lines.append("```")
    lines.append("\n## oracle (LLM-generated)")
    lines.append("```json")
    lines.append(json.dumps(result["oracle"], indent=2))
    lines.append("```")
    lines.append("\n## Failure")
    lines.append(f"stage:           **{result['failure_stage']}**")
    lines.append(f"failure_message: {result['failure_message']}")
    if result.get("verifier_report") is not None:
        vr = result["verifier_report"]
        lines.append(f"\nVerifierReport.reason:       **{vr['reason']}**")
        lines.append(f"VerifierReport.message:        {vr.get('message')}")
        lines.append(f"trajectory_steps:              {vr.get('trajectory_steps')}")
        lines.append(f"success_first_fired_at_step:   {vr.get('success_first_fired_at_step')}")
        if vr.get("terminal_state"):
            ts = vr["terminal_state"]
            lines.append(f"terminal ee_pos:               {ts.get('ee_pos')}")
            lines.append("terminal objects:")
            for name, obj in ts.get("objects", {}).items():
                lines.append(f"  {name}: pos={obj.get('pos')}")
    lines.append("\nEmit one patch via emit_patch.")
    return "\n".join(lines)


def _serialize_pipeline_result(result: Any) -> dict[str, Any]:
    """Convert a PipelineResult into a plain dict the prompt builder consumes."""
    out: dict[str, Any] = {
        "failure_stage": str(result.failure_stage) if result.failure_stage else None,
        "failure_message": result.failure_message,
        "task": result.task.model_dump() if result.task else None,
        "scene_code": result.scene_spec.scene_code if result.scene_spec else "",
        "success_code": result.success_spec.success_code if result.success_spec else "",
        "oracle": (
            result.success_spec.oracle.model_dump() if result.success_spec else {}
        ),
    }
    if result.verifier_report is not None:
        vr = result.verifier_report
        ts = vr.terminal_state
        out["verifier_report"] = {
            "reason": str(vr.reason),
            "message": vr.message,
            "trajectory_steps": len(vr.trajectory),
            "success_first_fired_at_step": vr.success_first_fired_at_step,
            "terminal_state": (
                None if ts is None else {
                    "ee_pos": ts.ee_pos.tolist(),
                    "ee_quat": ts.ee_quat.tolist(),
                    "objects": {
                        name: {
                            "pos": obj.pos.tolist(),
                            "quat": obj.quat.tolist(),
                        }
                        for name, obj in ts.objects.items()
                    },
                }
            ),
        }
    return out


# ---------- Public entrypoint ----------

def critique(
    result: Any,
    client: Anthropic | None = None,
    model: str = SONNET,
) -> tuple[CritiquePatch, TokenUsage]:
    """Diagnose a failed PipelineResult; return a structured patch + token usage."""
    client = client or get_client()
    payload_for_prompt = _serialize_pipeline_result(result)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        tools=[get_tool_schema()],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": _build_user_prompt(payload_for_prompt)}],
    )
    payload = extract_tool_input(resp, TOOL_NAME)
    patch = _validate_patch(payload)
    usage = TokenUsage(
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return patch, usage
