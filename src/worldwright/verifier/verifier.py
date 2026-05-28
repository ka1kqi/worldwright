"""Verifier: take a Solver-executed trajectory + LLM-emitted success() predicate,
return a structured pass/fail report.

The Verifier does NOT execute the oracle directly -- it calls into
``worldwright.solver.execute_oracle`` and then evaluates the success predicate.
This keeps the solver mechanical and reusable, and the verifier focused on
judgement + diagnostics for the future Critic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from worldwright.engine import FrankaHandle, SceneState, WorldwrightScene
from worldwright.solver import OracleResult, TrajectoryStep, execute_oracle
from worldwright.task import OracleHint


class VerdictReason(StrEnum):
    PASSED = "passed"
    SUCCESS_NEVER_FIRED = "success_never_fired"    # terminal state failed success()
    SUCCESS_FN_RAISED = "success_fn_raised"
    ORACLE_FAILED = "oracle_failed"
    EMPTY_TRAJECTORY = "empty_trajectory"


@dataclass
class VerifierReport:
    passed: bool
    reason: VerdictReason
    message: str | None
    trajectory: list[TrajectoryStep]
    terminal_state: SceneState | None
    success_first_fired_at_step: int | None
    oracle_result: OracleResult


def verify(
    scene: WorldwrightScene,
    franka: FrankaHandle,
    success_fn: Callable[[SceneState], bool],
    oracle: OracleHint,
    max_steps: int = 5000,
) -> VerifierReport:
    """Run the oracle, evaluate success() per step + at terminal, return a report."""
    oracle_result = execute_oracle(scene, franka, oracle, max_steps=max_steps)
    trajectory = oracle_result.trajectory
    terminal_state = trajectory[-1].state if trajectory else None

    # Diagnostic: did success() ever fire?
    first_fire: int | None = None
    for i, step in enumerate(trajectory):
        try:
            if success_fn(step.state):
                first_fire = i
                break
        except Exception as e:
            return VerifierReport(
                passed=False,
                reason=VerdictReason.SUCCESS_FN_RAISED,
                message=f"step {i}: {type(e).__name__}: {e}",
                trajectory=trajectory,
                terminal_state=terminal_state,
                success_first_fired_at_step=None,
                oracle_result=oracle_result,
            )

    if not oracle_result.success:
        return VerifierReport(
            passed=False,
            reason=VerdictReason.ORACLE_FAILED,
            message=(
                f"phase {oracle_result.last_phase_idx} "
                f"({oracle_result.failure_reason}): "
                f"{oracle_result.failure_message}"
            ),
            trajectory=trajectory,
            terminal_state=terminal_state,
            success_first_fired_at_step=first_fire,
            oracle_result=oracle_result,
        )

    if terminal_state is None:
        return VerifierReport(
            passed=False,
            reason=VerdictReason.EMPTY_TRAJECTORY,
            message="oracle produced no trajectory steps",
            trajectory=trajectory,
            terminal_state=None,
            success_first_fired_at_step=None,
            oracle_result=oracle_result,
        )

    # Verdict = success() at terminal state.
    try:
        terminal_ok = success_fn(terminal_state)
    except Exception as e:
        return VerifierReport(
            passed=False,
            reason=VerdictReason.SUCCESS_FN_RAISED,
            message=f"terminal: {type(e).__name__}: {e}",
            trajectory=trajectory,
            terminal_state=terminal_state,
            success_first_fired_at_step=first_fire,
            oracle_result=oracle_result,
        )

    if not terminal_ok:
        return VerifierReport(
            passed=False,
            reason=VerdictReason.SUCCESS_NEVER_FIRED,
            message="success() returned False at terminal state",
            trajectory=trajectory,
            terminal_state=terminal_state,
            success_first_fired_at_step=first_fire,
            oracle_result=oracle_result,
        )

    return VerifierReport(
        passed=True,
        reason=VerdictReason.PASSED,
        message=None,
        trajectory=trajectory,
        terminal_state=terminal_state,
        success_first_fired_at_step=first_fire,
        oracle_result=oracle_result,
    )
