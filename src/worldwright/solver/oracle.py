"""Oracle executor: take an OracleHint plan, drive the scene, return a trajectory.

The solver is mechanical: it has no judgement about whether the task succeeded
(that is the Verifier's job). It only translates each declarative ``OraclePhase``
into wrapper calls + ``scene.step()`` loops, recording a TrajectoryStep per step.

Failure modes are structured (``FailureReason``) so the Critic can act on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

import numpy as np

from worldwright.engine import FrankaHandle, SceneState, WorldwrightScene
from worldwright.task import OracleHint, OraclePhase


# ---------- defaults (used when an OraclePhase leaves them unset) ----------

DEFAULT_STEPS: dict[str, int] = {
    "ik_reach": 100,
    "ik_lift": 200,
    "ik_place": 200,
    "grasp": 100,
    "wait": 100,
}
DEFAULT_FORCE: float = -0.5
DEFAULT_WAYPOINTS: int = 200
PRE_GRASP_SETTLE_STEPS: int = 50
PLACE_RELEASE_STEPS: int = 50


# ---------- types ----------

class FailureReason(StrEnum):
    IK_INFEASIBLE = "ik_infeasible"
    PATH_NOT_FOUND = "path_not_found"
    PHASE_MISSING_PARAMS = "phase_missing_params"
    UNKNOWN_PHASE_TYPE = "unknown_phase_type"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    PHASE_EXECUTION_ERROR = "phase_execution_error"


@dataclass
class TrajectoryStep:
    t: float
    state: SceneState
    action: np.ndarray   # 9-vec joint position target most recently commanded
    phase_idx: int
    phase_type: str


@dataclass
class OracleResult:
    success: bool                                # all phases completed cleanly
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    last_phase_idx: int = -1                     # index of last attempted phase
    failure_reason: FailureReason | None = None
    failure_message: str | None = None


class _PhaseError(Exception):
    """Internal: raise from phase impl to abort the run with a structured reason."""

    def __init__(self, reason: FailureReason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


# ---------- helpers ----------

def _require_pos(phase: OraclePhase, phase_idx: int) -> tuple[float, float, float]:
    if phase.pos is None:
        raise _PhaseError(
            FailureReason.PHASE_MISSING_PARAMS,
            f"phase {phase_idx} ({phase.type}) requires pos",
        )
    return phase.pos


def _safe_ik(
    franka: FrankaHandle,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] | None,
    phase_idx: int,
) -> np.ndarray:
    try:
        q = franka.ik(pos, quat)
    except Exception as e:
        raise _PhaseError(
            FailureReason.IK_INFEASIBLE,
            f"phase {phase_idx}: ik raised {type(e).__name__}: {e}",
        ) from e
    if np.any(np.isnan(q)):
        raise _PhaseError(
            FailureReason.IK_INFEASIBLE,
            f"phase {phase_idx}: ik returned NaN",
        )
    return q


def _safe_plan(
    franka: FrankaHandle,
    q_goal: np.ndarray,
    n_waypoints: int,
    phase_idx: int,
) -> list[np.ndarray]:
    try:
        path = franka.plan(q_goal, n_waypoints=n_waypoints)
    except Exception as e:
        raise _PhaseError(
            FailureReason.PATH_NOT_FOUND,
            f"phase {phase_idx}: plan_path raised {type(e).__name__}: {e}",
        ) from e
    if not path:
        raise _PhaseError(
            FailureReason.PATH_NOT_FOUND,
            f"phase {phase_idx}: plan_path returned empty path",
        )
    return path


# ---------- phase implementations ----------

def _run_phase(
    scene: WorldwrightScene,
    franka: FrankaHandle,
    phase: OraclePhase,
    phase_idx: int,
    running_target: np.ndarray,
    record: Callable[[np.ndarray, int, str], None],
) -> None:
    t = phase.type

    if t == "ik_pre_grasp":
        pos = _require_pos(phase, phase_idx)
        q = _safe_ik(franka, pos, phase.quat, phase_idx)
        q[-2:] = franka.GRIPPER_OPEN_Q
        path = _safe_plan(franka, q, phase.n_waypoints or DEFAULT_WAYPOINTS, phase_idx)
        for wp in path:
            running_target[:] = wp
            franka.move_to(wp)
            scene.step()
            record(running_target, phase_idx, t)
        for _ in range(PRE_GRASP_SETTLE_STEPS):
            scene.step()
            record(running_target, phase_idx, t)
        return

    if t in ("ik_reach", "ik_lift", "ik_place"):
        pos = _require_pos(phase, phase_idx)
        q = _safe_ik(franka, pos, phase.quat, phase_idx)
        running_target[:7] = q[:7]
        franka.move_arm_to(q)
        n = phase.steps if phase.steps is not None else DEFAULT_STEPS[t]
        for _ in range(n):
            scene.step()
            record(running_target, phase_idx, t)
        if t == "ik_place":
            running_target[-2:] = franka.GRIPPER_OPEN_Q
            franka.open_gripper()
            for _ in range(PLACE_RELEASE_STEPS):
                scene.step()
                record(running_target, phase_idx, t)
        return

    if t == "grasp":
        force = phase.force if phase.force is not None else DEFAULT_FORCE
        franka.close_gripper(force)
        running_target[-2:] = 0.0   # log as "closed" position target
        n = phase.steps if phase.steps is not None else DEFAULT_STEPS[t]
        for _ in range(n):
            scene.step()
            record(running_target, phase_idx, t)
        return

    if t == "wait":
        n = phase.steps if phase.steps is not None else DEFAULT_STEPS[t]
        for _ in range(n):
            scene.step()
            record(running_target, phase_idx, t)
        return

    raise _PhaseError(
        FailureReason.UNKNOWN_PHASE_TYPE,
        f"phase {phase_idx}: unknown phase type {t!r}",
    )


# ---------- top-level ----------

def execute_oracle(
    scene: WorldwrightScene,
    franka: FrankaHandle,
    hint: OracleHint,
    max_steps: int = 5000,
) -> OracleResult:
    """Drive the scene through every phase in ``hint``, recording state per step."""
    trajectory: list[TrajectoryStep] = []
    running_target = np.zeros(9, dtype=float)

    def record(action: np.ndarray, phase_idx: int, phase_type: str) -> None:
        trajectory.append(
            TrajectoryStep(
                t=scene.t,
                state=scene.state(),
                action=action.copy(),
                phase_idx=phase_idx,
                phase_type=phase_type,
            )
        )

    for phase_idx, phase in enumerate(hint.phases):
        if len(trajectory) >= max_steps:
            return OracleResult(
                success=False, trajectory=trajectory, last_phase_idx=phase_idx,
                failure_reason=FailureReason.MAX_STEPS_EXCEEDED,
                failure_message=f"exceeded max_steps={max_steps}",
            )
        try:
            _run_phase(scene, franka, phase, phase_idx, running_target, record)
        except _PhaseError as e:
            return OracleResult(
                success=False, trajectory=trajectory, last_phase_idx=phase_idx,
                failure_reason=e.reason, failure_message=e.message,
            )
        except Exception as e:
            return OracleResult(
                success=False, trajectory=trajectory, last_phase_idx=phase_idx,
                failure_reason=FailureReason.PHASE_EXECUTION_ERROR,
                failure_message=f"{type(e).__name__}: {e}",
            )

    return OracleResult(
        success=True, trajectory=trajectory,
        last_phase_idx=len(hint.phases) - 1,
    )
