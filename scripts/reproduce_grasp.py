"""Manual reproduction of examples/tutorials/IK_motion_planning_grasp.py with a success assertion.

This is the human-baseline vertical-slice target: the exact pick-and-lift our generative
pipeline must eventually re-derive. Run from repo root with the venv active:

    .venv/bin/python scripts/reproduce_grasp.py [--backend metal|cpu] [--viewer]
                                                [--record-mp4 path.mp4] [--fps 60]

Exits 0 on success (cube lifted to >= 0.15 m and held by gripper), non-zero otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np


CUBE_HALF = 0.02
CUBE_INIT_POS = (0.65, 0.0, CUBE_HALF)
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
PRE_GRASP_Z = 0.25
REACH_Z = 0.130
LIFT_Z = 0.28


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def run(backend_name: str, show_viewer: bool, record_mp4: str | None, fps: int) -> bool:
    import genesis as gs

    backend = {"metal": gs.metal, "cpu": gs.cpu, "gpu": gs.gpu}[backend_name]
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(2 * CUBE_HALF,) * 3, pos=CUBE_INIT_POS)
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
    )

    cam = None
    if record_mp4 is not None:
        cam = scene.add_camera(
            res=(960, 720),
            pos=(2.2, -1.4, 1.2),
            lookat=(0.65, 0.0, 0.15),
            fov=35,
            GUI=False,
        )

    t0 = time.time()
    scene.build()
    print(f"[build] {time.time() - t0:.1f}s", flush=True)

    if cam is not None:
        cam.start_recording()

    def step():
        scene.step()
        if cam is not None:
            cam.render()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")

    # 1) pre-grasp via planned path
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, PRE_GRASP_Z]),
        quat=GRASP_QUAT,
    )
    qpos[-2:] = 0.04  # fingers open
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=200)
    for wp in path:
        franka.control_dofs_position(wp)
        step()
    # settle at pre-grasp
    for _ in range(100):
        step()

    # 2) reach
    qpos = franka.inverse_kinematics(
        link=end_effector, pos=np.array([0.65, 0.0, REACH_Z]), quat=GRASP_QUAT,
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for _ in range(100):
        step()

    # 3) grasp (close fingers via force control, hold arm)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
    for _ in range(100):
        step()

    # 4) lift
    qpos = franka.inverse_kinematics(
        link=end_effector, pos=np.array([0.65, 0.0, LIFT_Z]), quat=GRASP_QUAT,
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for _ in range(200):
        step()

    if cam is not None:
        cam.stop_recording(save_to_filename=record_mp4, fps=fps)
        print(f"[mp4]    {record_mp4}", flush=True)

    cube_pos = to_np(cube.get_pos())
    ee_pos = to_np(end_effector.get_pos())
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
    p.add_argument("--record-mp4", default=None, help="write a rendered mp4 to this path")
    p.add_argument("--fps", type=int, default=60, help="output mp4 framerate")
    args = p.parse_args()
    ok = run(args.backend, args.viewer, args.record_mp4, args.fps)
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
