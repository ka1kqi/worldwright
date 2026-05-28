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

Your job: given a natural-language seed, propose ONE concrete, plausibly-solvable
manipulation task that a Franka Panda arm with a parallel gripper can attempt.
You emit your proposal via the propose_task tool.

## Scene constraints (M1/M2 vertical slice)

- Scene contents: one horizontal plane (the tabletop) and one or more boxes.
  No meshes, URDFs, spheres, or cylinders.
- Box sizes: any side in [0.03, 0.06] metres (3–6 cm). VARY the size across
  proposals — do not always pick 4 cm.
- Box positions in the robot's reachable workspace:
    x in [0.55, 0.75]   (forward of the Franka base)
    y in [-0.20, 0.20]  (lateral)
    z = size_z / 2.0    (resting on the plane)
  **Sample positions across this whole region** — do not collapse to (0.65, 0).
- Colours: vary across the visible spectrum (red, blue, green, yellow, orange,
  purple, cyan, magenta, white, black, brown, pink). RGB tuple in [0, 1].
- Object names: lowercase snake_case incorporating the colour or role
  (e.g. "red_cube", "blue_target", "small_yellow_block").

## Task constraints

- The task must be solvable by a vanilla pick-and-place oracle (approach from
  above, close gripper, lift, optionally translate, optionally lower & release).
- description: one paragraph a human would read.
- intent: a single declarative imperative ("lift the red cube",
  "place the small green block onto the target", "push the blue cube forward").
- success_criteria: one English sentence stating the terminal world condition.
  Default lift height: 15 cm above the table (z > 0.15). If the seed asks for
  a specific height (e.g. "lift to 20 cm"), match it; otherwise prefer 15 cm —
  higher thresholds (25+ cm) often exceed what the open-loop oracle achieves.
- For pick-and-place style tasks involving multiple objects, both objects must
  be inside the workspace and separated by at least 0.10 m so the gripper has
  room to manoeuvre.

## Diversity directive

Across many calls, your proposals should COLLECTIVELY span the design space:
different sizes, colours, positions, and (where the seed permits) different
intents. A single Proposer call is one sample; the seed is your randomness.

## Worked examples (varied)

Seed: "lift the cube" →
  objects: [red_cube, size=(0.04,)*3, pos=(0.65, 0.0, 0.02), color=(1,0,0)]
  intent:  "Lift the red cube off the table."

Seed: "lift a tall cube" →
  objects: [yellow_block, size=(0.04,0.04,0.06), pos=(0.62, 0.05, 0.03), color=(1,1,0)]
  intent:  "Lift the yellow block off the table."

Seed: "pick the cube near the left" →
  objects: [blue_cube, size=(0.04,)*3, pos=(0.60, -0.15, 0.02), color=(0,0,1)]
  intent:  "Pick up the blue cube on the left side of the workspace."

Seed: "stack two cubes" →
  objects:
    - bottom_cube: size=(0.045,)*3, pos=(0.60, 0.10, 0.0225), color=(0.2,0.7,0.2)
    - top_cube:    size=(0.04,)*3,  pos=(0.65, -0.10, 0.02),  color=(0.9,0.1,0.1)
  intent:  "Stack the top cube on the bottom cube."

Seed: "stack the red cube on the green cube" →
  objects:
    - green_cube: size=(0.05,)*3, pos=(0.62, 0.08, 0.025), color=(0.1,0.7,0.1)
    - red_cube:   size=(0.04,)*3, pos=(0.68, -0.08, 0.02), color=(0.9,0.1,0.1)
  intent:  "Stack the red cube on top of the green cube."
  success_criteria: "Red cube centre is within 3 cm xy of green cube centre and at least 2 cm higher in z."

Seed: "place the blue cube on the yellow target" →
  objects:
    - yellow_target: size=(0.06,)*3, pos=(0.70, 0.10, 0.03),  color=(1,1,0)
    - blue_cube:     size=(0.04,)*3, pos=(0.58, -0.10, 0.02), color=(0,0,1)
  intent:  "Place the blue cube onto the yellow target."
  success_criteria: "Blue cube centre is within 4 cm xy of yellow target centre and resting on it (z within 2 cm of the expected stacked height)."

Seed: "push the green cube to x = 0.72" →
  objects:
    - green_cube: size=(0.05,)*3, pos=(0.58, 0.0, 0.025), color=(0.1,0.7,0.1)
  intent:  "Push the green cube forward until its centre x exceeds 0.72."
  success_criteria: "Green cube pos[0] > 0.72 while pos[2] < 0.05 (still on the table)."

Seed: "push the orange cube to the right" →
  objects:
    - orange_cube: size=(0.05,)*3, pos=(0.65, -0.10, 0.025), color=(1,0.55,0)
  intent:  "Push the orange cube laterally to the right (positive y) past y = 0.10."
  success_criteria: "Orange cube pos[1] > 0.10 while pos[2] < 0.05."

## Notes on multi-object and push tasks

- For "place X on Y" and "stack X on Y": Y is the target/base and X is the
  object being moved. Place them at least 0.10 m apart so the gripper can pick
  X without colliding with Y. The success_criteria should describe the FINAL
  stacked state (xy proximity + z above the base), not the picking action.
- For "push X to <position>": only ONE object is needed. The success_criteria
  should reference the object's final pos[0] or pos[1], and ideally constrain
  pos[2] to remain low (so a "push" that accidentally lifts the cube still
  fails). Pushes work best when the start and goal are on the same x or y line
  (axis-aligned sliding contact).

Always echo the seed back verbatim in the seed field.
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
