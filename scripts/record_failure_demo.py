"""Deterministic reproduction of the M1.5 LLM-failure mode.

The LLM RewardCoder generated an oracle whose reach z worked for a 4 cm cube
but not for the 5 cm cube the Proposer + SceneCoder produced. The gripper
descended too far relative to the cube's centre and nudged the cube aside
during grasp, leading to a clean Verifier failure with
``reason=success_never_fired``.

This script reproduces exactly that failure mode deterministically (no LLM
call) and records it to mp4, so we have a portfolio-ready artifact showing
the structured failure surface that the Critic (M2) consumes.

    .venv/bin/python scripts/record_failure_demo.py [--record-mp4 PATH] [--fps 60]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from worldwright.engine import WorldwrightScene
from worldwright.task import OracleHint, OraclePhase
from worldwright.utils import compile_callable
from worldwright.verifier import verify


# 5 cm cube — the size the LLM Proposer settled on.
CUBE_NAME = "red_cube"
CUBE_POS = (0.6, 0.0, 0.025)
CUBE_SIZE = (0.05, 0.05, 0.05)


def build_5cm_cube_scene(scene: WorldwrightScene) -> None:
    scene.add_plane()
    scene.add_franka()
    scene.add_box(
        name=CUBE_NAME, size=CUBE_SIZE, pos=CUBE_POS, color=(1.0, 0.1, 0.1)
    )


# Bad oracle: reach z fits a 4 cm cube (top at z=0.04), not a 5 cm cube
# (top at z=0.05). Same z values RewardCoder copy-pasted from Example A.
BAD_ORACLE = OracleHint(
    target_object=CUBE_NAME,
    phases=[
        OraclePhase(type="ik_pre_grasp", pos=(0.6, 0.0, 0.25),  n_waypoints=200),
        OraclePhase(type="ik_reach",     pos=(0.6, 0.0, 0.13),  steps=100),
        OraclePhase(type="grasp",        force=-0.5,            steps=100),
        OraclePhase(type="ik_lift",      pos=(0.6, 0.0, 0.28),  steps=200),
    ],
)


SUCCESS_SRC = """
import numpy as np

def success(state):
    cube = state.objects.get("red_cube")
    if cube is None:
        return False
    ee = state.ee_pos
    xy_err = float(np.hypot(cube.pos[0] - ee[0], cube.pos[1] - ee[1]))
    return bool(cube.pos[2] > 0.15 and xy_err < 0.05)
"""


def run(backend: str, record_mp4: str | None, fps: int) -> bool:
    success_fn = compile_callable(
        SUCCESS_SRC, "success", extra_globals={"np": np}
    )

    with WorldwrightScene(sim_dt=0.01, backend=backend) as scene:
        build_5cm_cube_scene(scene)

        if record_mp4 is not None:
            scene.add_camera(
                name="cam0", pos=(2.2, -1.4, 1.2),
                lookat=(0.65, 0.0, 0.15), res=(960, 720), fov=35,
            )

        scene.build()

        if record_mp4 is not None:
            scene.start_recording("cam0")

        init = scene.state()
        print(f"[build] cube={init.objects[CUBE_NAME].pos.round(3).tolist()}")

        t0 = time.time()
        report = verify(scene, scene.franka, success_fn, BAD_ORACLE)
        print(f"[verify] {time.time() - t0:.1f}s  "
              f"passed={report.passed} reason={report.reason}  "
              f"steps={len(report.trajectory)}  "
              f"first_fire={report.success_first_fired_at_step}")
        if report.message:
            print(f"[message] {report.message}")
        if report.terminal_state is not None:
            term = report.terminal_state
            print(f"  terminal cube: pos={term.objects[CUBE_NAME].pos.round(3).tolist()}")
            print(f"  terminal ee:   pos={term.ee_pos.round(3).tolist()}")

        if record_mp4 is not None:
            scene.stop_recording("cam0", save_to_filename=record_mp4, fps=fps)
            print(f"[mp4] {record_mp4}")

    # This script EXPECTS failure -- the demo is the failure itself.
    print("[expected: FAIL]", "as expected" if not report.passed else "UNEXPECTED PASS")
    return not report.passed  # success == we demonstrated the failure


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    p.add_argument("--record-mp4", default=None)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()
    ok = run(args.backend, args.record_mp4, args.fps)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
