"""RewardCoder agent — TaskSpec → SuccessSpec (success() source + OracleHint).

Opus 4.7 per DESIGN.html §4.1 — this is the highest-stakes agent in the pipeline.
A subtly wrong success predicate poisons every downstream episode.

The agent returns BOTH the success-function source AND the scripted oracle plan
in a single tool call, because they must agree on object names, target poses,
and numeric thresholds — splitting them risks drift.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from worldwright.task import SuccessSpec, TaskSpec
from worldwright.task.spec import TokenUsage

from ._anthropic import OPUS, extract_tool_input, get_client


TOOL_NAME = "emit_reward"
MAX_TOKENS = 3072

SYSTEM = """You are a robotics reward designer. Given a TaskSpec, produce TWO things in a single
emit_reward tool call:

1. **success_code** — Python source for `def success(state) -> bool` that returns True iff the
   task is complete in the given world state.
2. **oracle** — a scripted plan the solver will execute to verify the task is solvable.

## The state argument

`state` is a SceneState dataclass with these attributes:

    state.t              : float                       sim time in seconds
    state.franka_q       : np.ndarray (9,)             [arm0..arm6, finger0, finger1]
    state.franka_qdot    : np.ndarray (9,)
    state.ee_pos         : np.ndarray (3,)             end-effector world position
    state.ee_quat        : np.ndarray (4,)             end-effector orientation (w, x, y, z)
    state.objects        : dict[str, ObjectState]      keyed by object name
        ObjectState.pos  : np.ndarray (3,)
        ObjectState.quat : np.ndarray (4,)
    state.contacts       : list[ContactInfo]           currently empty in M1; do not rely on

## success() rules

- Function name: exactly `success`. Argument: exactly `state`. Returns: `bool`.
- May `import numpy as np` and `import math`. No other imports.
- Do NOT raise exceptions; return False on edge cases (missing object, NaN, etc).
- Use object names exactly as they appear in TaskSpec.objects.
- Map the TaskSpec's success_criteria language to numeric thresholds in metres
  (e.g. "15 cm above the table" → `obj.pos[2] > 0.15`).
- Default xy proximity threshold for "held by gripper" checks: 0.05 m.

## OracleHint rules

A list of `phases` executed in order. Each phase has:

    type            — one of: ik_pre_grasp, ik_reach, grasp, ik_lift, ik_place, wait
    pos             — (x, y, z) IK target (for ik_* phases)
    quat            — (w, x, y, z) gripper orientation; default (0, 1, 0, 0) is gripper-down
    force           — finger force for grasp; -0.5 is the canonical close force
    steps           — number of sim steps to hold the command
    n_waypoints     — waypoints for ik_pre_grasp (uses motion planner)
    target_object   — which object the phase manipulates

Top-level `target_object` is the primary object being acted on.

## Worked examples

### Example A — "Lift the red cube 15 cm above the table"
Cube starts at z=0.025 (so its centre is 2.5 cm above the table — a 5 cm cube
half is 2.5 cm). Reach depth is chosen so the gripper fingers wrap around the
**middle** of the cube. For an object centred at cube.pos with cube_size on each
side, use:

    pre_grasp z = cube.pos[2] + 0.225            # ~22 cm above the centre
    reach z     = cube.pos[2] + 0.105            # gripper at object centre + 10.5 cm
    lift z      = cube.pos[2] + 0.255            # final centre height

These are PARAMETRIC — do not hard-code 0.13 / 0.25 / 0.28. Compute them from
the TaskSpec object's pos[2].

success_code:
```python
import numpy as np
def success(state):
    cube = state.objects.get("red_cube")
    if cube is None:
        return False
    ee = state.ee_pos
    xy_err = float(np.hypot(cube.pos[0] - ee[0], cube.pos[1] - ee[1]))
    return bool(cube.pos[2] > 0.15 and xy_err < 0.05)
```

For a cube at pos=(0.65, 0.0, 0.02) (4 cm cube), the oracle would be:
```json
{
  "target_object": "red_cube",
  "phases": [
    {"type": "ik_pre_grasp", "pos": [0.65, 0.0, 0.245], "n_waypoints": 200},
    {"type": "ik_reach",     "pos": [0.65, 0.0, 0.125], "steps": 100},
    {"type": "grasp",        "force": -0.5,             "steps": 100},
    {"type": "ik_lift",      "pos": [0.65, 0.0, 0.275], "steps": 200}
  ]
}
```

For a cube at pos=(0.6, 0.0, 0.025) (5 cm cube), the oracle would be:
```json
{
  "target_object": "red_cube",
  "phases": [
    {"type": "ik_pre_grasp", "pos": [0.6, 0.0, 0.250], "n_waypoints": 200},
    {"type": "ik_reach",     "pos": [0.6, 0.0, 0.130], "steps": 100},
    {"type": "grasp",        "force": -0.5,            "steps": 100},
    {"type": "ik_lift",      "pos": [0.6, 0.0, 0.280], "steps": 200}
  ]
}
```

Notice how the reach z **tracks the object centre** so the gripper fingers can
wrap symmetrically around the object regardless of object size.

### Example B — "Push the blue cube past x = 0.7"
Sliding contact task; gripper closed, low approach:

success_code:
```python
def success(state):
    cube = state.objects.get("blue_cube")
    if cube is None:
        return False
    return bool(cube.pos[0] > 0.70)
```

oracle:
```json
{
  "target_object": "blue_cube",
  "phases": [
    {"type": "ik_pre_grasp", "pos": [0.50, 0.0, 0.10], "n_waypoints": 200},
    {"type": "grasp",        "force": -0.5,            "steps": 50},
    {"type": "ik_reach",     "pos": [0.55, 0.0, 0.04], "steps": 100},
    {"type": "ik_reach",     "pos": [0.75, 0.0, 0.04], "steps": 200}
  ]
}
```

### Example C — "Stack the small red cube on top of the green cube"

success_code:
```python
import numpy as np
def success(state):
    red = state.objects.get("red_cube")
    green = state.objects.get("green_cube")
    if red is None or green is None:
        return False
    xy_err = float(np.hypot(red.pos[0] - green.pos[0], red.pos[1] - green.pos[1]))
    z_above = float(red.pos[2] - green.pos[2])
    return bool(xy_err < 0.03 and z_above > 0.02)
```

oracle is two pick-and-place cycles. Always include n_waypoints on the FIRST
ik_pre_grasp of each cycle; subsequent IK phases can use steps alone.

## Quality checklist before emitting

- Did I read TaskSpec.success_criteria carefully? Numeric thresholds match the text.
- Does success() reference the right object names from TaskSpec.objects?
- Does the oracle's target_object exist in TaskSpec.objects?
- Are oracle phase positions inside the workspace (x in [0.4, 0.8], y in [-0.3, 0.3])?
- **Did I compute reach_z from the object's pos[2], not hard-code 0.13?** Different
  object sizes need different reach depths.
- Does the oracle actually drive the world toward the success() == True condition?
"""


def get_tool_schema() -> dict[str, Any]:
    """Tool input_schema is the SuccessSpec JSON schema."""
    schema = SuccessSpec.model_json_schema()
    return {
        "name": TOOL_NAME,
        "description": (
            "Emit success() source + the scripted OracleHint. Both must agree on "
            "object names, target poses, and thresholds."
        ),
        "input_schema": schema,
    }


def _user_prompt(task: TaskSpec) -> str:
    return (
        "Design success() + oracle for this task.\n\n"
        f"TaskSpec JSON:\n{task.model_dump_json(indent=2)}"
    )


def emit_reward(
    task: TaskSpec,
    client: Anthropic | None = None,
    model: str = OPUS,
) -> tuple[SuccessSpec, TokenUsage]:
    client = client or get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        tools=[get_tool_schema()],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": _user_prompt(task)}],
    )
    payload = extract_tool_input(resp, TOOL_NAME)
    spec = SuccessSpec.model_validate(payload)
    usage = TokenUsage(
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return spec, usage
