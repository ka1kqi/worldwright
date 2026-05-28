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
    reach z     = cube.pos[2] + 0.085            # gripper FINGERTIPS at cube centre - small offset
    lift z      = cube.pos[2] + 0.255            # final centre height

These are PARAMETRIC — do not hard-code 0.13 / 0.25 / 0.28. Compute them from
the TaskSpec object's pos[2].

**Mandatory grasp parameters:** force = -1.0 (not -0.5) and steps = 200 (not
100). Stronger force + more settling time is the difference between catching
the cube and nudging it sideways. Additionally, ALWAYS insert a `wait` phase
of 50 steps between the grasp and the ik_lift, to let the grasp fully settle
before the arm starts moving upward.

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
    {"type": "ik_reach",     "pos": [0.65, 0.0, 0.105], "steps": 100},
    {"type": "grasp",        "force": -1.0,             "steps": 200},
    {"type": "wait",         "steps": 50},
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
    {"type": "ik_reach",     "pos": [0.6, 0.0, 0.110], "steps": 100},
    {"type": "grasp",        "force": -1.0,            "steps": 200},
    {"type": "wait",         "steps": 50},
    {"type": "ik_lift",      "pos": [0.6, 0.0, 0.280], "steps": 200}
  ]
}
```

Notice how reach z **tracks the object centre** so the gripper fingertips wrap
symmetrically around the object regardless of object size, and how the grasp
phase uses force=-1.0 + steps=200 + a 50-step wait before lifting.

### Example B — "Push the blue cube past x = 0.7"
There is NO dedicated `push` phase type. Compose a push from the existing
primitives: approach behind the cube with the gripper closed, then use one or
more `ik_reach` phases to translate the closed gripper through the cube
along the desired axis. Keep the reach z just above the cube's centre so the
gripper face contacts the side of the cube, not the top.

For a cube at pos=(0.55, 0.0, 0.025) (5 cm cube) pushing to x>0.70:

success_code:
```python
def success(state):
    cube = state.objects.get("blue_cube")
    if cube is None:
        return False
    return bool(cube.pos[0] > 0.70 and cube.pos[2] < 0.05)
```

oracle:
```json
{
  "target_object": "blue_cube",
  "phases": [
    {"type": "grasp",        "force": -1.0,             "steps": 80},
    {"type": "ik_pre_grasp", "pos": [0.45, 0.0, 0.15],  "n_waypoints": 200},
    {"type": "ik_reach",     "pos": [0.45, 0.0, 0.035], "steps": 120},
    {"type": "ik_reach",     "pos": [0.78, 0.0, 0.035], "steps": 300}
  ]
}
```

Notes for push: (1) close the gripper FIRST so the fingers act as a flat
pushing face; (2) start the approach behind the cube along the push axis
(here x≈0.45, behind a cube at x=0.55); (3) keep z low (just above the
table) during the slide; (4) overshoot the goal slightly in the final reach
so the cube actually crosses the threshold.

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

For red_cube=(0.68, -0.08, 0.02) (4 cm) and green_cube=(0.62, 0.08, 0.025)
(5 cm), the oracle is a single pick-and-place cycle that picks `red_cube`
and places it on top of `green_cube`. Target xy = green_cube.pos[:2]; target
z = green_cube.pos[2] + green_size/2 + red_size/2 + small_clearance.

oracle:
```json
{
  "target_object": "red_cube",
  "phases": [
    {"type": "ik_pre_grasp", "pos": [0.68, -0.08, 0.245], "n_waypoints": 200},
    {"type": "ik_reach",     "pos": [0.68, -0.08, 0.105], "steps": 100},
    {"type": "grasp",        "force": -1.0,               "steps": 200},
    {"type": "wait",         "steps": 50},
    {"type": "ik_lift",      "pos": [0.68, -0.08, 0.275], "steps": 200},
    {"type": "ik_reach",     "pos": [0.62,  0.08, 0.275], "steps": 200},
    {"type": "ik_place",     "pos": [0.62,  0.08, 0.115], "steps": 200}
  ]
}
```

The `ik_place` phase opens the gripper at the end (release), letting the red
cube settle on top of the green cube. Compute target z parametrically from
both cube sizes — do not hard-code 0.115.

### Example D — "Place the blue cube onto the yellow target"

This is structurally identical to Example C: a single pick-and-place cycle
that grasps the moved object and releases it above the target object. The
only differences from a stack are (a) the target object may be wider, so
xy proximity tolerance can be looser, and (b) the success check is usually
"on top of the target" rather than the strict 3 cm xy of a stack.

success_code:
```python
import numpy as np
def success(state):
    cube = state.objects.get("blue_cube")
    target = state.objects.get("yellow_target")
    if cube is None or target is None:
        return False
    xy_err = float(np.hypot(cube.pos[0] - target.pos[0], cube.pos[1] - target.pos[1]))
    z_above = float(cube.pos[2] - target.pos[2])
    return bool(xy_err < 0.04 and z_above > 0.015)
```

The oracle follows the Example C template — pick blue_cube, lift, traverse to
the yellow_target's xy, then ik_place at z = target.pos[2] + target_size/2 +
cube_size/2 + clearance.

oracle is two pick-and-place cycles for stacks of three or more; for a single
"place X on Y" it is exactly one cycle. Always include n_waypoints on the
FIRST ik_pre_grasp of each cycle; subsequent IK phases can use steps alone.

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
