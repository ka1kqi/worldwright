"""SceneCoder agent — TaskSpec → Python source for build_scene(scene).

Sonnet 4.6. Output is sandbox-audited before exec; the wrapper API is the only
surface the LLM is told about, so it can't reach into ``genesis`` directly.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from worldwright.task import SceneSpec, TaskSpec
from worldwright.task.spec import TokenUsage

from ._anthropic import SONNET, extract_tool_input, get_client


TOOL_NAME = "emit_scene_code"
MAX_TOKENS = 2048

# Surface we expose to the LLM. Keep this PRECISE — anything missing is a hole.
SCENE_API_DOC = """\
The function receives one argument: `scene`, a WorldwrightScene. Available methods:

    scene.add_plane(name: str = "plane") -> EntityHandle
        Add the horizontal ground plane at z=0.

    scene.add_box(
        name: str,
        size: tuple[float, float, float],          # FULL size in metres (not half)
        pos:  tuple[float, float, float],          # (x, y, z) of the centre
        color: tuple[float, float, float] | None = None,   # (r, g, b) in [0, 1]
    ) -> EntityHandle
        Add a rigid box.

    scene.add_franka(name: str = "franka") -> FrankaHandle
        Add the Franka Panda at the origin. Always call this exactly once.

You must NOT call scene.build(), scene.step(), or any other method.
You must NOT import any module other than `numpy` (as np) and `math`.
"""

SYSTEM = f"""You are a robot-simulation programmer.

Given a TaskSpec, write the Python source for a function

    def build_scene(scene) -> None: ...

that uses the worldwright engine wrapper to construct the described scene.

{SCENE_API_DOC}

Rules:
- Define exactly one top-level function named build_scene.
- Always add the plane first, the Franka second, then each object from the TaskSpec.
- Object positions: place the object centre at pos = (x, y, size_z / 2.0) so the
  object rests on the plane. The TaskSpec already gives reasonable positions; you may
  use them verbatim or override z to satisfy the resting condition.
- No assertions, no print statements, no side effects beyond the scene calls.
- Output your code via the emit_scene_code tool. Do not wrap it in markdown fences.

## Worked examples

### Single object — "lift the red cube"
```python
def build_scene(scene):
    scene.add_plane()
    scene.add_franka()
    scene.add_box(
        name="red_cube",
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
        color=(1.0, 0.0, 0.0),
    )
```

### Multi-object stack/place — "stack the red cube on the green cube"
Two boxes from TaskSpec.objects, each rested on the plane. Loop or unrolled is
fine; either way you MUST emit every object from the TaskSpec.
```python
def build_scene(scene):
    scene.add_plane()
    scene.add_franka()
    for obj in [
        ("green_cube", (0.05, 0.05, 0.05), (0.62,  0.08, 0.025), (0.1, 0.7, 0.1)),
        ("red_cube",   (0.04, 0.04, 0.04), (0.68, -0.08, 0.02),  (0.9, 0.1, 0.1)),
    ]:
        name, size, pos, color = obj
        scene.add_box(name=name, size=size, pos=pos, color=color)
```

### Push task — "push the green cube to x = 0.72"
A push needs only one object. Place it at the START position the TaskSpec
specifies; do not pre-translate it toward the goal.
```python
def build_scene(scene):
    scene.add_plane()
    scene.add_franka()
    scene.add_box(
        name="green_cube",
        size=(0.05, 0.05, 0.05),
        pos=(0.58, 0.0, 0.025),
        color=(0.1, 0.7, 0.1),
    )
```

For "place X on Y" tasks the layout is identical to the stack example: both
objects start on the plane, separated, and the oracle (not the scene) moves X
onto Y. Never pre-stack the objects in build_scene.
"""


def get_tool_schema() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Emit the full Python source of the build_scene function.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scene_code": {
                    "type": "string",
                    "description": (
                        "Full Python source defining `def build_scene(scene): ...`. "
                        "No markdown fences, no preamble."
                    ),
                },
            },
            "required": ["scene_code"],
        },
    }


def _user_prompt(task: TaskSpec) -> str:
    return (
        "Implement build_scene for the following task.\n\n"
        f"TaskSpec JSON:\n{task.model_dump_json(indent=2)}"
    )


def emit_scene(
    task: TaskSpec,
    client: Anthropic | None = None,
    model: str = SONNET,
) -> tuple[SceneSpec, TokenUsage]:
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
    spec = SceneSpec(scene_code=payload["scene_code"])
    usage = TokenUsage(
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return spec, usage
