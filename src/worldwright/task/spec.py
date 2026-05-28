"""Pydantic schemas the agents pass between each other.

Wire shape (M1):
    Proposer     →  TaskSpec
    SceneCoder   →  SceneSpec     (build_scene source)
    RewardCoder  →  SuccessSpec   (success() source + OracleHint)
    Pipeline     →  TaskManifest  (full record persisted to disk)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Proposer output ----------

class ObjectSpec(BaseModel):
    """One object in the proposed scene. The Proposer describes; the SceneCoder realises."""
    name: str
    type: Literal["box", "sphere", "cylinder", "mesh", "urdf"]
    pos: tuple[float, float, float]
    size: tuple[float, ...] | None = None         # box: (sx, sy, sz); sphere/cylinder: (r[, h])
    color: tuple[float, float, float] | None = None
    file: str | None = None                       # mesh / urdf path


class TaskSpec(BaseModel):
    """Natural-language + structured task description from the Proposer."""
    seed: str
    description: str                              # one paragraph, what a human would read
    domain: str = "franka_tabletop_manipulation"
    intent: str                                   # declarative one-liner ("lift the red cube")
    objects: list[ObjectSpec]
    success_criteria: str                         # NL; RewardCoder turns this into success()


# ---------- SceneCoder output ----------

class SceneSpec(BaseModel):
    """SceneCoder output: source for `def build_scene(scene: WorldwrightScene) -> None`."""
    scene_code: str


# ---------- RewardCoder output ----------

OraclePhaseType = Literal[
    "ik_pre_grasp",   # IK to a pose above the target with gripper open
    "ik_reach",       # IK down to grasp depth
    "grasp",          # close gripper with force control
    "ik_lift",        # IK upward
    "ik_place",       # IK to a placement pose, then open gripper
    "wait",           # settle (no command change)
]


class OraclePhase(BaseModel):
    type: OraclePhaseType
    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None
    force: float | None = None
    steps: int | None = None
    n_waypoints: int | None = None                # for plan_path-using phases
    target_object: str | None = None              # for grasp / place phases


class OracleHint(BaseModel):
    """Scripted plan the Solver will execute to test solvability."""
    target_object: str                            # primary object being manipulated
    phases: list[OraclePhase]


class SuccessSpec(BaseModel):
    """RewardCoder output: success() source + the oracle plan."""
    success_code: str                             # `def success(state: SceneState) -> bool`
    oracle: OracleHint


# ---------- Per-task metrics ----------

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class TaskMetrics(BaseModel):
    critic_iterations: int = 0
    tokens: dict[str, TokenUsage] = Field(default_factory=dict)  # keyed by agent name
    wallclock_s: float = 0.0
    oracle_steps: int = 0


# ---------- Full persisted record ----------

class TaskManifest(BaseModel):
    """Full record of one validated task. Serialized to task_manifest.json."""
    task_id: str
    task: TaskSpec
    scene: SceneSpec
    success: SuccessSpec
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    genesis_version: str | None = None
    git_sha: str | None = None
