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
        size: tuple[float, float, float],          # (sx, sy, sz) in metres
        pos:  tuple[float, float, float],          # (x, y, z) of centre
        color: tuple[float, float, float] | None = None,   # (r, g, b) 0-1
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
