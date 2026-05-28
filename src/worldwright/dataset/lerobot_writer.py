"""Persist a validated task to disk in three forms:

1. **LeRobot episode** -- HuggingFace's modern format, the headline deliverable.
2. **Raw NPZ shadow** -- flat dict-of-arrays for fast in-repo debugging
   without a LeRobot dependency on read.
3. **Task manifest JSON** -- full TaskManifest pydantic record.

All three live under ``data/<dataset_name>/`` so a validated task has one place
to look up everything about itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldwright.solver import TrajectoryStep
from worldwright.task import (
    SceneSpec,
    SuccessSpec,
    TaskManifest,
    TaskMetrics,
    TaskSpec,
)


SIM_FPS = 100   # dt = 0.01 in WorldwrightScene
ROBOT_TYPE = "franka_panda"


# ---------------------------------------------------------------------------
# Feature schema (LeRobot) -- adapts to the objects in the TaskSpec.
# ---------------------------------------------------------------------------

_ARM_JOINT_NAMES = [f"arm.q{i}" for i in range(7)]
_GRIPPER_JOINT_NAMES = ["gripper.q0", "gripper.q1"]
_FRANKA_DOF_NAMES = _ARM_JOINT_NAMES + _GRIPPER_JOINT_NAMES
_POSE7_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def _features_for_task(task: TaskSpec) -> dict[str, dict]:
    # LeRobot's validator compares numpy.shape (tuple) to feature["shape"]
    # with !=, so shapes MUST be tuples or validation fails.
    feats: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(_FRANKA_DOF_NAMES),
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": list(_POSE7_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(_FRANKA_DOF_NAMES),
        },
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for obj in task.objects:
        feats[f"observation.objects.{obj.name}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": list(_POSE7_NAMES),
        }
    return feats


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _frame_from_step(
    step: TrajectoryStep,
    task: TaskSpec,
    is_done: bool,
) -> dict:
    state = step.state
    frame: dict = {
        "observation.state": state.franka_q.astype(np.float32),
        "observation.ee_pose": np.concatenate(
            [state.ee_pos, state.ee_quat]
        ).astype(np.float32),
        "action": step.action.astype(np.float32),
        "reward": np.array([1.0 if is_done else 0.0], dtype=np.float32),
        "next.done": np.array([is_done], dtype=bool),
        # LeRobot 0.5+ requires the natural-language task on every frame.
        "task": task.intent,
    }
    for obj in task.objects:
        os = state.objects.get(obj.name)
        if os is None:
            frame[f"observation.objects.{obj.name}"] = np.zeros(7, dtype=np.float32)
        else:
            frame[f"observation.objects.{obj.name}"] = np.concatenate(
                [os.pos, os.quat]
            ).astype(np.float32)
    return frame


def write_lerobot_episode(
    root: Path,
    repo_id: str,
    task: TaskSpec,
    trajectory: list[TrajectoryStep],
    success_first_fired_at_step: int | None,
) -> None:
    """Append one episode to a LeRobot dataset (creating the dataset on first call)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    features = _features_for_task(task)

    info_path = root / "meta" / "info.json"
    if info_path.exists():
        ds = LeRobotDataset(repo_id=repo_id, root=root)
    else:
        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=SIM_FPS,
            features=features,
            root=root,
            robot_type=ROBOT_TYPE,
            use_videos=False,
        )

    for i, step in enumerate(trajectory):
        is_done = (
            success_first_fired_at_step is not None
            and i >= success_first_fired_at_step
        )
        frame = _frame_from_step(step, task, is_done)
        ds.add_frame(frame)

    ds.save_episode()


def write_npz_shadow(
    path: Path,
    task: TaskSpec,
    trajectory: list[TrajectoryStep],
    success_first_fired_at_step: int | None,
) -> None:
    """Flat dict-of-arrays NPZ for in-repo debugging (no LeRobot dep on read)."""
    if not trajectory:
        raise ValueError("cannot write NPZ for empty trajectory")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, np.ndarray] = {
        "t":              np.array([s.t for s in trajectory], dtype=np.float32),
        "franka_q":       np.stack([s.state.franka_q    for s in trajectory]).astype(np.float32),
        "franka_qdot":    np.stack([s.state.franka_qdot for s in trajectory]).astype(np.float32),
        "ee_pos":         np.stack([s.state.ee_pos      for s in trajectory]).astype(np.float32),
        "ee_quat":        np.stack([s.state.ee_quat     for s in trajectory]).astype(np.float32),
        "action":         np.stack([s.action            for s in trajectory]).astype(np.float32),
        "phase_idx":      np.array([s.phase_idx for s in trajectory], dtype=np.int32),
        "phase_type":     np.array([s.phase_type for s in trajectory], dtype="S20"),
    }
    for obj in task.objects:
        poses = []
        quats = []
        for s in trajectory:
            os = s.state.objects.get(obj.name)
            poses.append(os.pos if os is not None else np.zeros(3))
            quats.append(os.quat if os is not None else np.array([1.0, 0.0, 0.0, 0.0]))
        data[f"obj.{obj.name}.pos"] = np.stack(poses).astype(np.float32)
        data[f"obj.{obj.name}.quat"] = np.stack(quats).astype(np.float32)

    if success_first_fired_at_step is not None:
        data["success_first_fired_at_step"] = np.array(
            [success_first_fired_at_step], dtype=np.int32
        )

    np.savez_compressed(path, **data)


def write_task_manifest(path: Path, manifest: TaskManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def write_validated_task(
    *,
    dataset_root: Path | str,
    repo_id: str,
    task_id: str,
    task: TaskSpec,
    scene: SceneSpec,
    success: SuccessSpec,
    trajectory: list[TrajectoryStep],
    success_first_fired_at_step: int | None,
    metrics: TaskMetrics | None = None,
    genesis_version: str | None = None,
) -> dict[str, str]:
    """Persist a validated task: LeRobot episode + NPZ shadow + JSON manifest.

    Returns a dict of the on-disk paths created.
    """
    dataset_root = Path(dataset_root)

    write_lerobot_episode(
        root=dataset_root,
        repo_id=repo_id,
        task=task,
        trajectory=trajectory,
        success_first_fired_at_step=success_first_fired_at_step,
    )

    npz_path = dataset_root / "raw" / f"{task_id}.npz"
    write_npz_shadow(
        path=npz_path,
        task=task,
        trajectory=trajectory,
        success_first_fired_at_step=success_first_fired_at_step,
    )

    manifest = TaskManifest(
        task_id=task_id,
        task=task,
        scene=scene,
        success=success,
        metrics=metrics or TaskMetrics(),
        genesis_version=genesis_version,
    )
    manifest_path = dataset_root / "manifests" / f"{task_id}.json"
    write_task_manifest(manifest_path, manifest)

    return {
        "lerobot_root":   str(dataset_root),
        "npz_path":       str(npz_path),
        "manifest_path":  str(manifest_path),
    }
