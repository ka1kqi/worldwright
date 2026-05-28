"""M1.5 acceptance gate -- Solver + Verifier on a hardcoded TaskSpec, OracleHint,
and success() predicate. Uses the known-good baseline scene + plan that
scripts/reproduce_grasp_wrapped.py already proves works.

This isolates whether the Solver + Verifier *mechanism* works, independent of
LLM-output quality (the LLM-end-to-end version is scripts/smoke_m1_5.py).

    .venv/bin/python scripts/smoke_m1_5_hardcoded.py [--backend metal]

Acceptance (issue #6): passed=True with terminal cube z >= 0.15.
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


# Baseline scene: 4 cm cube at (0.65, 0.0, 0.02). Known to be grasp-friendly.
CUBE_NAME = "cube"
CUBE_POS = (0.65, 0.0, 0.02)
CUBE_SIZE = (0.04, 0.04, 0.04)


def build_baseline_scene(scene: WorldwrightScene) -> None:
    scene.add_plane()
    scene.add_franka()
    scene.add_box(name=CUBE_NAME, size=CUBE_SIZE, pos=CUBE_POS)


BASELINE_ORACLE = OracleHint(
    target_object=CUBE_NAME,
    phases=[
        OraclePhase(type="ik_pre_grasp", pos=(0.65, 0.0, 0.25), n_waypoints=200),
        OraclePhase(type="ik_reach",     pos=(0.65, 0.0, 0.13), steps=100),
        OraclePhase(type="grasp",        force=-0.5,            steps=100),
        OraclePhase(type="ik_lift",      pos=(0.65, 0.0, 0.28), steps=200),
    ],
)


BASELINE_SUCCESS_SRC = """
import numpy as np

def success(state):
    cube = state.objects.get("cube")
    if cube is None:
        return False
    ee = state.ee_pos
    xy_err = float(np.hypot(cube.pos[0] - ee[0], cube.pos[1] - ee[1]))
    return bool(cube.pos[2] > 0.15 and xy_err < 0.05)
"""


def run(backend: str) -> bool:
    success_fn = compile_callable(
        BASELINE_SUCCESS_SRC, "success", extra_globals={"np": np}
    )

    with WorldwrightScene(sim_dt=0.01, backend=backend) as scene:
        build_baseline_scene(scene)
        scene.build()
        init = scene.state()
        print(f"[build] cube={init.objects[CUBE_NAME].pos.round(3).tolist()}  "
              f"ee={init.ee_pos.round(3).tolist()}")

        t0 = time.time()
        report = verify(scene, scene.franka, success_fn, BASELINE_ORACLE)
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

    return report.passed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    args = p.parse_args()
    ok = run(args.backend)
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
