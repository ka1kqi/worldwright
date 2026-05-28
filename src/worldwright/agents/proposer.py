"""Proposer agent — seed string → TaskSpec.

Sonnet 4.6 via tool_use. Outputs a TaskSpec whose JSON schema is generated
from the Pydantic model so the agent and the validator can never drift apart.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from worldwright.task import TaskSpec
from worldwright.task.spec import TokenUsage

from ._anthropic import SONNET, extract_tool_input, get_client


TOOL_NAME = "propose_task"
MAX_TOKENS = 2048

SYSTEM = """You are a robotics task designer for the Franka tabletop manipulation domain.

Your job: given a natural-language seed, propose ONE simple, obviously-solvable
manipulation task that a Franka Panda arm with a parallel gripper can attempt.
You emit your proposal via the propose_task tool.

Constraints for the M1 vertical slice:
- Scene contents are limited to a single horizontal plane (the tabletop) and one
  or more boxes. No meshes, no URDFs, no spheres, no cylinders.
- Each box must sit in the robot's reachable workspace:
    x in [0.40, 0.80]  (forward)
    y in [-0.30, 0.30] (lateral)
    z = (size_z / 2.0) (resting on the plane)
- Sizes: cubes between 0.03 m and 0.06 m on each side.
- The task must be solvable by a vanilla pick-and-place oracle: approach from
  above, close gripper, lift, optionally translate, optionally lower and release.
- Object names: lowercase snake_case (e.g., "red_cube", "green_target").
- description: one paragraph a human would read.
- intent: a single declarative imperative ("lift the red cube").
- success_criteria: one English sentence stating the terminal world condition
  (e.g., "the red cube is at least 15 cm above the table and aligned with the
  gripper").
- Always echo the seed back verbatim in the seed field.
"""


def get_tool_schema() -> dict[str, Any]:
    """JSON schema for the propose_task tool, generated from the Pydantic model."""
    schema = TaskSpec.model_json_schema()
    # Anthropic tool input_schema must be an object; Pydantic gives us that.
    return {
        "name": TOOL_NAME,
        "description": (
            "Emit a structured TaskSpec proposal. Fill every required field. "
            "Echo the user-provided seed back verbatim."
        ),
        "input_schema": schema,
    }


def _user_prompt(seed: str) -> str:
    return (
        f"Seed: {seed!r}\n"
        f"Domain: franka_tabletop_manipulation\n\n"
        f"Propose ONE task that satisfies the constraints in the system prompt."
    )


def propose(
    seed: str,
    client: Anthropic | None = None,
    model: str = SONNET,
) -> tuple[TaskSpec, TokenUsage]:
    """Generate a TaskSpec for ``seed``. Returns (spec, token_usage)."""
    client = client or get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        tools=[get_tool_schema()],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": _user_prompt(seed)}],
    )
    payload = extract_tool_input(resp, TOOL_NAME)
    spec = TaskSpec.model_validate(payload)
    # Guarantee the seed survives even if the model rewrote it.
    if spec.seed != seed:
        spec = spec.model_copy(update={"seed": seed})
    usage = TokenUsage(
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return spec, usage
