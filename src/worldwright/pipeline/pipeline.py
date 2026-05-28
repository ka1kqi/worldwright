"""Pipeline orchestrator -- run a single seed through every stage.

For M1 there is no Critic loop, so any stage failure aborts the attempt and
returns a structured ``PipelineResult``. The CLI may retry the whole pipeline
N times (each attempt is a fresh set of LLM calls).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from worldwright.agents import emit_reward, emit_scene, propose
from worldwright.dataset import write_validated_task
from worldwright.engine import WorldwrightScene
from worldwright.task import SceneSpec, SuccessSpec, TaskMetrics, TaskSpec
from worldwright.utils import SandboxViolation, compile_callable
from worldwright.verifier import VerifierReport, verify


StageName = Literal[
    "proposer",
    "scene_coder",
    "reward_coder",
    "sandbox",
    "scene_build",
    "verify",
    "dataset_write",
]


@dataclass
class PipelineResult:
    task_id: str
    seed: str
    success: bool                                  # task was validated AND written
    failure_stage: Optional[StageName] = None
    failure_message: Optional[str] = None
    paths: Optional[dict[str, str]] = None         # if success, on-disk artifacts
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    task: Optional[TaskSpec] = None
    scene_spec: Optional[SceneSpec] = None
    success_spec: Optional[SuccessSpec] = None
    verifier_report: Optional[VerifierReport] = None
    attempt: int = 1                               # 1-based attempt number


def _failure(
    task_id: str,
    seed: str,
    stage: StageName,
    e: BaseException,
    metrics: TaskMetrics,
    attempt: int,
    **carry,
) -> PipelineResult:
    return PipelineResult(
        task_id=task_id, seed=seed,
        success=False,
        failure_stage=stage,
        failure_message=f"{type(e).__name__}: {e}",
        metrics=metrics, attempt=attempt,
        **carry,
    )


def _new_task_id() -> str:
    return f"wright-{uuid.uuid4().hex[:10]}"


def run_one(
    seed: str,
    *,
    dataset_name: str = "vs-m1",
    backend: str = "metal",
    repo_id: str = "ka1kqi/worldwright-vs-m1",
    task_id: Optional[str] = None,
    show_viewer: bool = False,
    log=print,
    attempt: int = 1,
) -> PipelineResult:
    """Run the full pipeline on ``seed`` exactly once. No retries."""
    if task_id is None:
        task_id = _new_task_id()
    metrics = TaskMetrics()
    t_start = time.time()

    # ---- Proposer ----
    t0 = time.time()
    try:
        task, prop_usage = propose(seed)
    except Exception as e:
        return _failure(task_id, seed, "proposer", e, metrics, attempt)
    metrics.tokens["proposer"] = prop_usage
    log(f"[proposer]    {time.time() - t0:5.1f}s  "
        f"intent={task.intent!r}  objects={[o.name for o in task.objects]}")

    # ---- SceneCoder ----
    t0 = time.time()
    try:
        scene_spec, sc_usage = emit_scene(task)
    except Exception as e:
        return _failure(task_id, seed, "scene_coder", e, metrics, attempt, task=task)
    metrics.tokens["scene_coder"] = sc_usage
    log(f"[scene_coder] {time.time() - t0:5.1f}s")

    # ---- RewardCoder ----
    t0 = time.time()
    try:
        success_spec, rw_usage = emit_reward(task)
    except Exception as e:
        return _failure(task_id, seed, "reward_coder", e, metrics, attempt,
                        task=task, scene_spec=scene_spec)
    metrics.tokens["reward_coder"] = rw_usage
    log(f"[reward_coder]{time.time() - t0:5.1f}s  "
        f"oracle_phases={len(success_spec.oracle.phases)}  "
        f"target={success_spec.oracle.target_object!r}")

    # ---- Sandbox compile ----
    try:
        build_scene = compile_callable(
            scene_spec.scene_code, "build_scene", extra_globals={"np": np},
        )
        success_fn = compile_callable(
            success_spec.success_code, "success", extra_globals={"np": np},
        )
    except (SandboxViolation, KeyError, TypeError, SyntaxError) as e:
        return _failure(task_id, seed, "sandbox", e, metrics, attempt,
                        task=task, scene_spec=scene_spec, success_spec=success_spec)

    # ---- Scene build + Verify ----
    try:
        with WorldwrightScene(
            sim_dt=0.01, backend=backend, show_viewer=show_viewer
        ) as scene:
            build_scene(scene)
            scene.build()
            report = verify(scene, scene.franka, success_fn, success_spec.oracle)
    except Exception as e:
        return _failure(task_id, seed, "scene_build", e, metrics, attempt,
                        task=task, scene_spec=scene_spec, success_spec=success_spec)

    metrics.oracle_steps = len(report.trajectory)
    log(f"[verify]      {time.time() - t0:5.1f}s  "
        f"passed={report.passed}  reason={report.reason}  "
        f"steps={len(report.trajectory)}  "
        f"first_fire={report.success_first_fired_at_step}")

    if not report.passed:
        result = PipelineResult(
            task_id=task_id, seed=seed,
            success=False,
            failure_stage="verify",
            failure_message=f"{report.reason}: {report.message}",
            metrics=metrics, attempt=attempt,
            task=task, scene_spec=scene_spec, success_spec=success_spec,
            verifier_report=report,
        )
        metrics.wallclock_s = time.time() - t_start
        return result

    # ---- Dataset write ----
    t0 = time.time()
    metrics.wallclock_s = time.time() - t_start  # exclude write time? include? include.
    try:
        paths = write_validated_task(
            dataset_root=Path("data") / dataset_name,
            repo_id=repo_id,
            task_id=task_id,
            task=task, scene=scene_spec, success=success_spec,
            trajectory=report.trajectory,
            success_first_fired_at_step=report.success_first_fired_at_step,
            metrics=metrics, genesis_version="1.0.0",
        )
    except Exception as e:
        return _failure(task_id, seed, "dataset_write", e, metrics, attempt,
                        task=task, scene_spec=scene_spec, success_spec=success_spec,
                        verifier_report=report)
    log(f"[dataset]     {time.time() - t0:5.1f}s  wrote {paths['lerobot_root']}")

    metrics.wallclock_s = time.time() - t_start
    return PipelineResult(
        task_id=task_id, seed=seed,
        success=True,
        paths=paths,
        metrics=metrics, attempt=attempt,
        task=task, scene_spec=scene_spec, success_spec=success_spec,
        verifier_report=report,
    )


def run_with_retries(
    seed: str,
    *,
    max_attempts: int = 3,
    log=print,
    **kwargs,
) -> PipelineResult:
    """Retry ``run_one`` up to ``max_attempts`` on any non-validating outcome.

    Returns the first successful PipelineResult, or the last failure if every
    attempt fails. This is a placeholder for the real Critic loop landing in M2 --
    here we just re-roll the LLMs, hoping the non-determinism delivers a
    working oracle. Cost grows linearly: budget ~$0.10-0.15 per attempt.
    """
    last: PipelineResult | None = None
    for i in range(1, max_attempts + 1):
        log(f"\n[attempt] {i}/{max_attempts}")
        last = run_one(seed, attempt=i, log=log, **kwargs)
        if last.success:
            return last
        log(f"[attempt {i} failed] stage={last.failure_stage}  "
            f"msg={last.failure_message}")
    assert last is not None
    return last
