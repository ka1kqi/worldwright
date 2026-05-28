"""Wrapper-based vertical-slice reproduction.

Same physics result as scripts/reproduce_grasp.py, but driven through the
worldwright.engine wrapper instead of raw `genesis` calls. Serves as the
acceptance test for M1.1.

    .venv/bin/python scripts/reproduce_grasp_wrapped.py --backend metal
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from worldwright.engine import WorldwrightScene


CUBE_HALF = 0.02
CUBE_INIT_POS = (0.65, 0.0, CUBE_HALF)
PRE_GRASP_Z = 0.25
REACH_Z = 0.130
LIFT_Z = 0.28


def run(backend: str, show_viewer: bool, record_mp4: str | None, fps: int) -> bool:
    with WorldwrightScene(sim_dt=0.01, backend=backend, show_viewer=show_viewer) as scene:
        scene.add_plane()
        cube = scene.add_box(name="cube", size=(2 * CUBE_HALF,) * 3, pos=CUBE_INIT_POS)
        franka = scene.add_franka()

        if record_mp4 is not None:
            scene.add_camera(
                name="cam0", pos=(2.2, -1.4, 1.2),
                lookat=(0.65, 0.0, 0.15), res=(960, 720), fov=35,
            )

        t0 = time.time()
        scene.build()
        print(f"[build] {time.time() - t0:.1f}s", flush=True)

        if record_mp4 is not None:
            scene.start_recording("cam0")

        # 1) pre-grasp via planned path
        q_pre = franka.ik((0.65, 0.0, PRE_GRASP_Z))
        q_pre[-2:] = franka.GRIPPER_OPEN_Q
        for wp in franka.plan(q_pre, n_waypoints=200):
            franka.move_to(wp)
            scene.step()
        for _ in range(100):
            scene.step()

        # 2) reach
        q_reach = franka.ik((0.65, 0.0, REACH_Z))
        franka.move_arm_to(q_reach)
        for _ in range(100):
            scene.step()

        # 3) grasp — hold arm, close fingers under force control
        franka.move_arm_to(q_reach)
        franka.close_gripper(force=-0.5)
        for _ in range(100):
            scene.step()

        # 4) lift
        q_lift = franka.ik((0.65, 0.0, LIFT_Z))
        franka.move_arm_to(q_lift)
        for _ in range(200):
            scene.step()

        if record_mp4 is not None:
            scene.stop_recording("cam0", save_to_filename=record_mp4, fps=fps)
            print(f"[mp4]    {record_mp4}", flush=True)

        state = scene.state()
        cube_pos = state.objects["cube"].pos
        ee_pos = state.ee_pos
        xy_err = math.hypot(cube_pos[0] - ee_pos[0], cube_pos[1] - ee_pos[1])
        lifted = bool(cube_pos[2] > 0.15)
        held = xy_err < 0.05

        print(
            f"[result] cube_pos={cube_pos.round(3).tolist()}  "
            f"ee_pos={ee_pos.round(3).tolist()}  xy_err={xy_err:.3f}  "
            f"lifted={lifted} held={held}"
        )
        return lifted and held


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["metal", "cpu", "gpu"], default="metal")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--record-mp4", default=None)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()
    ok = run(args.backend, args.viewer, args.record_mp4, args.fps)
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
