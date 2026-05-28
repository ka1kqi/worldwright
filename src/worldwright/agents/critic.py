"""Critic agent -- failure-aware patcher.

Reads a failed PipelineResult and emits a structured patch:

    PatchOracle             — tweak only the oracle plan (most common)
    PatchSuccess            — re-emit success() + oracle (oracle also needs to move)
    PatchSuccessThreshold   — re-emit success() only; reuse the existing oracle
    PatchScene              — re-emit build_scene (scene malformed; rare)
    Unsolvable              — give up cleanly with a reason

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


class PatchSuccessThreshold(BaseModel):
    """Light-weight success() rewrite that keeps the oracle plan as-is.

    Use when the oracle already drives the cube to a physically sensible
    state, but the success() thresholds (z cutoff, xy_err) are overstated
    relative to what the oracle plan can actually achieve.
    """
    kind: Literal["patch_success_threshold"] = "patch_success_threshold"
    success_code: str
    rationale: str


class PatchScene(BaseModel):
    kind: Literal["patch_scene"] = "patch_scene"
    scene_code: str
    rationale: str


class Unsolvable(BaseModel):
    kind: Literal["unsolvable"] = "unsolvable"
    reason: str


CritiquePatch = Union[
    PatchOracle, PatchSuccess, PatchSuccessThreshold, PatchScene, Unsolvable,
]


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
                    "enum": [
                        "patch_oracle",
                        "patch_success",
                        "patch_success_threshold",
                        "patch_scene",
                        "unsolvable",
                    ],
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
                    "description": (
                        "New def success(state) source. Required for patch_success "
                        "AND patch_success_threshold."
                    ),
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
    if kind == "patch_success_threshold":
        return PatchSuccessThreshold(
            success_code=payload["success_code"], rationale=rationale,
        )
    if kind == "patch_scene":
        return PatchScene(scene_code=payload["scene_code"], rationale=rationale)
    if kind == "unsolvable":
        return Unsolvable(reason=payload.get("reason", rationale))
    raise ValueError(f"unknown patch_kind: {kind!r}")


# ---------- Prompt ----------

SYSTEM = """You are the Critic. The worldwright pipeline tried to validate a task and FAILED.
Your job: diagnose the failure and emit ONE structured patch via the emit_patch tool.

You must choose ONE of five patch kinds:

    patch_oracle             Tweak only the OracleHint (most common). Reuse
                             the same success_code; just adjust pos / quat /
                             force / steps in one or more phases.

    patch_success            Re-emit BOTH success() and oracle. Use this when
                             BOTH the threshold AND the oracle plan need to
                             move (e.g. you want to land the cube at a
                             different xy, AND relax the xy_err threshold).

    patch_success_threshold  Re-emit ONLY success(); keep the existing oracle
                             verbatim. Use this when the oracle already
                             drives the cube to a physically sensible state,
                             but the success thresholds are overstated
                             relative to what's actually achievable (e.g.
                             demands z > 0.20 but oracle peaks at z = 0.18,
                             or demands xy_err < 0.03 but the cube drifts
                             0.04 m during grasp+lift). This is the lightest
                             possible fix when the *measurement* is the
                             problem, not the *plan*.

    patch_scene              Re-emit only build_scene. Use this when the
                             scene_code crashed during build or produced an
                             obviously broken scene (objects overlapping the
                             robot, etc.). Rare.

    unsolvable               The task as described is fundamentally not
                             achievable with our M1 Franka+gripper setup.
                             Give a one-paragraph reason. Use sparingly.

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
     -> LIFT FELL SHORT. First try patch_oracle: increase ik_lift target z to
        threshold + 0.05. BUT if the ik_lift target z is already at/near the
        physical ceiling (~0.30 m for Franka in tabletop config) OR the lift
        target was already at success_threshold + 0.05 and still fell short,
        the threshold itself is overstated -> patch_success_threshold: lower
        the z cutoff to (cube_terminal_z - 0.005) so it matches reality.
   - cube_terminal_z >= success threshold by a lot
     -> success() criteria itself wrong (e.g. xy_err threshold too tight).
        patch_success_threshold: relax the success_code thresholds (xy_err,
        gripper-distance gate), keep oracle unchanged.
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

## Failure-pattern decision tree

These are the three high-frequency M2 failure modes observed in the
vs-m2 batch (10/86 verify-stage failures). Match the symptom to the
prescribed patch BEFORE falling back to a generic patch_oracle.

A. **Overstated success threshold** (e.g. "raise the cube to 20 cm",
   "lift the cube to 15 cm", "hoist the cube").
   - Symptom: cube_terminal_z is consistently *above* the cube's initial_z
     by >= 0.10 m (i.e. the grasp + lift physically worked) but is BELOW the
     numeric z cutoff hard-coded in success_code. The oracle's own ik_lift
     target z is at or above the success threshold; the cube just doesn't
     quite reach it because of finger compliance / IK error.
   - Fix: patch_success_threshold. Lower the z cutoff in success() to
     (cube_terminal_z - 0.005). Do NOT also bump the oracle -- the oracle is
     already trying its best.

B. **Edge-of-workspace IK** (e.g. "lift a cube far back", "lift a cube in
   the front", "lift a cube on the left/right side").
   - Symptom: cube_terminal_z ~ initial_z (cube barely moved) AND the
     cube's initial xy is at the periphery of the Franka workspace:
     x < 0.45 m or x > 0.75 m, |y| > 0.25 m. The ik_pre_grasp / ik_reach
     phases either ran but produced near-zero ee motion (IK clamped) or
     the reach pose was unreachable. ee_pos in terminal_state ends up far
     from the cube (xy distance > 0.05 m).
   - Fix: patch_oracle. Move the pre_grasp / reach / lift targets to xy
     coordinates that are within the safe workspace box
     (x in [0.50, 0.70], y in [-0.20, 0.20]) AND ALSO change the lift
     destination to the same in-workspace xy so the cube ends up where the
     success() check expects. If success() pins on the original cube xy,
     also relax xy_err in success_code via patch_success.
     If the cube itself is unreachable (initial pos outside the box),
     escalate to unsolvable with reason "task position outside Franka
     workspace box (x,y)=(...)".

C. **Small-cube 3 cm grasp** (e.g. "lift the small ... cube",
   "grasp the small white/black cube", "pick up the small red cube").
   - Symptom: cube has size ~0.03 m (note: success of small cubes at 5 cm
     is fine; the dangerous case is when size <= 0.035). cube_terminal_z
     remains within +/- 0.005 m of initial_z (grasp slipped). ee_pos in
     terminal_state is roughly where the cube was, but the cube did not
     come along.
   - Fix: patch_oracle, two-pronged:
       (1) ik_reach.pos[2] should be initial_cube_z + half_size - 0.008
           (i.e. fingertips dip BELOW the cube's vertical centre so the
           cube sits between the pads, not pinched at the top edge).
       (2) grasp.force = -1.5 (was -0.5 / -1.0), grasp.steps = 200, and
           ADD a wait phase of 80-100 steps right after the grasp so the
           cube settles into the pads before ik_lift starts pulling.
     Do NOT relax success() for small-cube failures -- the issue is grip,
     not measurement.

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
