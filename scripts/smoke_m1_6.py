"""M1.6 acceptance gate -- run hardcoded baseline through Solver+Verifier,
write the resulting trajectory to LeRobot + NPZ + manifest, then read back the
LeRobot episode to confirm it loads cleanly.

    .venv/bin/python scripts/smoke_m1_6.py [--backend metal] [--dataset-name vs-m1]

Acceptance (issue #7):
    - LeRobotDataset(<path>) loads the episode without error
    - NPZ shadow has matching shapes
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from worldwright.dataset import write_validated_task
from worldwright.engine import WorldwrightScene
from worldwright.task import (
    ObjectSpec,
    SceneSpec,
    SuccessSpec,
    TaskMetrics,
    TaskSpec,
)
from worldwright.utils import compile_callable
from worldwright.verifier import verify


def _import_baseline():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smoke_m1_5_hardcoded import (
        BASELINE_ORACLE,
        BASELINE_SUCCESS_SRC,
        CUBE_NAME,
        CUBE_POS,
        CUBE_SIZE,
        build_baseline_scene,
    )
    return (BASELINE_ORACLE, BASELINE_SUCCESS_SRC, CUBE_NAME, CUBE_POS,
            CUBE_SIZE, build_baseline_scene)


def run(backend: str, dataset_name: str) -> bool:
    (BASELINE_ORACLE, BASELINE_SUCCESS_SRC, CUBE_NAME, CUBE_POS,
     CUBE_SIZE, build_baseline_scene) = _import_baseline()

    dataset_root = Path("data") / dataset_name
    if dataset_root.exists():
        print(f"[clean] removing existing {dataset_root}")
        shutil.rmtree(dataset_root)

    task = TaskSpec(
        seed="vs-m1-baseline",
        description=(
            "Pick up the 4 cm red cube from the table and lift it at least "
            "15 cm above the surface, held centred under the gripper."
        ),
        intent="lift the red cube",
        objects=[
            ObjectSpec(name=CUBE_NAME, type="box", pos=CUBE_POS,
                       size=CUBE_SIZE, color=(1.0, 0.1, 0.1)),
        ],
        success_criteria="cube z > 0.15 m AND xy_err to gripper < 0.05 m",
    )
    scene_spec = SceneSpec(scene_code=(
        "def build_scene(scene):\n"
        "    scene.add_plane()\n"
        "    scene.add_franka()\n"
        f"    scene.add_box(name='{CUBE_NAME}', size={tuple(CUBE_SIZE)}, "
        f"pos={tuple(CUBE_POS)}, color=(1.0, 0.1, 0.1))\n"
    ))
    success_spec = SuccessSpec(success_code=BASELINE_SUCCESS_SRC,
                                oracle=BASELINE_ORACLE)
    success_fn = compile_callable(
        success_spec.success_code, "success", extra_globals={"np": np}
    )

    print(f"[run] backend={backend} dataset={dataset_root}")

    with WorldwrightScene(sim_dt=0.01, backend=backend) as scene:
        build_baseline_scene(scene)
        scene.build()
        t0 = time.time()
        report = verify(scene, scene.franka, success_fn, success_spec.oracle)
        print(f"[verify] {time.time() - t0:.1f}s  "
              f"passed={report.passed} reason={report.reason}  "
              f"steps={len(report.trajectory)}  "
              f"first_fire={report.success_first_fired_at_step}")

    if not report.passed:
        print(f"[abort] baseline verify failed: {report.message}")
        return False

    metrics = TaskMetrics(
        critic_iterations=0, wallclock_s=0.0,
        oracle_steps=len(report.trajectory),
    )

    t0 = time.time()
    paths = write_validated_task(
        dataset_root=dataset_root,
        repo_id="ka1kqi/worldwright-vs-m1",
        task_id="wright-0000001",
        task=task, scene=scene_spec, success=success_spec,
        trajectory=report.trajectory,
        success_first_fired_at_step=report.success_first_fired_at_step,
        metrics=metrics, genesis_version="1.0.0",
    )
    print(f"[write] {time.time() - t0:.1f}s")
    for k, v in paths.items():
        print(f"   {k}: {v}")

    # Acceptance 1: LeRobot dataset reload
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    loaded = LeRobotDataset(repo_id="ka1kqi/worldwright-vs-m1", root=dataset_root)
    n_eps = loaded.num_episodes
    n_frames = loaded.num_frames
    print(f"[reload] episodes={n_eps} frames={n_frames}")
    if n_eps != 1 or n_frames != len(report.trajectory):
        print(f"[FAIL] episode/frame count mismatch "
              f"(expected 1 / {len(report.trajectory)})")
        return False

    sample = loaded[0]
    print(f"[reload] sample keys: {sorted(sample.keys())[:10]}...")
    expected = {"observation.state", "observation.ee_pose", "action",
                "reward", "next.done", f"observation.objects.{CUBE_NAME}"}
    missing = expected - set(sample.keys())
    if missing:
        print(f"[FAIL] missing keys in reload: {missing}")
        return False

    # Acceptance 2: NPZ shadow shapes
    npz = np.load(paths["npz_path"])
    print(f"[npz] keys: {sorted(npz.files)}")
    n = len(report.trajectory)
    expected_shapes = {
        "t": (n,),
        "franka_q": (n, 9),
        "ee_pos": (n, 3),
        "ee_quat": (n, 4),
        "action": (n, 9),
        f"obj.{CUBE_NAME}.pos": (n, 3),
        f"obj.{CUBE_NAME}.quat": (n, 4),
    }
    for k, want in expected_shapes.items():
        got = npz[k].shape
        if got != want:
            print(f"[FAIL] npz[{k}] shape {got} != expected {want}")
            return False
        print(f"   npz[{k}] {got} OK")

    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    p.add_argument("--dataset-name", default="vs-m1")
    args = p.parse_args()
    ok = run(args.backend, args.dataset_name)
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
