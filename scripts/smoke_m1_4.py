"""M1.4 smoke test — Proposer + RewardCoder; verify success() behaviour on
synthesized initial and lifted states without running the full sim.

    .venv/bin/python scripts/smoke_m1_4.py [--seed "lift the cube"]

Acceptance (from issue #5):
    - success(initial_state)  is False
    - success(lifted_state)   is True
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from worldwright.agents import emit_reward, propose
from worldwright.engine import ObjectState, SceneState
from worldwright.utils import compile_callable


def _initial_state(cube_name: str, cube_xy: tuple[float, float]) -> SceneState:
    """Mock SceneState matching the moment after scene.build(): cube resting on table,
    end-effector parked at the Franka neutral pose well above + behind the workspace.
    """
    return SceneState(
        t=0.0,
        franka_q=np.zeros(9),
        franka_qdot=np.zeros(9),
        ee_pos=np.array([0.3, 0.0, 0.6]),
        ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        objects={
            cube_name: ObjectState(
                pos=np.array([cube_xy[0], cube_xy[1], 0.025]),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        },
        contacts=[],
    )


def _lifted_state(cube_name: str, cube_xy: tuple[float, float]) -> SceneState:
    """Mock SceneState matching a successful lift: cube at z=0.20, ee directly above."""
    return SceneState(
        t=10.0,
        franka_q=np.zeros(9),
        franka_qdot=np.zeros(9),
        ee_pos=np.array([cube_xy[0], cube_xy[1], 0.28]),
        ee_quat=np.array([0.0, 1.0, 0.0, 0.0]),
        objects={
            cube_name: ObjectState(
                pos=np.array([cube_xy[0], cube_xy[1], 0.20]),
                quat=np.array([0.0, 1.0, 0.0, 0.0]),
            ),
        },
        contacts=[],
    )


def run(seed: str) -> bool:
    # 1. Proposer
    t0 = time.time()
    task, prop_usage = propose(seed)
    print(f"[proposer] {time.time() - t0:.1f}s  "
          f"tokens in={prop_usage.input_tokens} out={prop_usage.output_tokens}")
    print(f"[task] intent={task.intent!r}  objects={[o.name for o in task.objects]}")

    boxes = [o for o in task.objects if o.type == "box"]
    if not boxes:
        print("[FAIL] no box in TaskSpec")
        return False
    target = boxes[0]

    # 2. RewardCoder
    t0 = time.time()
    success_spec, reward_usage = emit_reward(task)
    print(f"[reward_coder] {time.time() - t0:.1f}s  "
          f"tokens in={reward_usage.input_tokens} out={reward_usage.output_tokens}")
    print(f"[success_code]\n{success_spec.success_code}")
    print(f"[oracle] target_object={success_spec.oracle.target_object}")
    for i, ph in enumerate(success_spec.oracle.phases):
        print(f"  phase {i}: {ph.type} "
              f"pos={ph.pos} force={ph.force} steps={ph.steps} "
              f"n_waypoints={ph.n_waypoints}")

    # 3. Sandbox-compile success()
    success_fn = compile_callable(
        success_spec.success_code, "success", extra_globals={"np": np},
    )

    # 4. Acceptance: initial → False, lifted → True
    init = _initial_state(target.name, (target.pos[0], target.pos[1]))
    lift = _lifted_state(target.name, (target.pos[0], target.pos[1]))

    init_result = success_fn(init)
    lift_result = success_fn(lift)
    print(f"[success(initial)] {init_result}")
    print(f"[success(lifted)]  {lift_result}")

    ok = (init_result is False) and (lift_result is True)
    if not ok:
        print("[FAIL] success() did not discriminate initial vs lifted")
        return False

    # 5. Oracle sanity: must mention the target object
    if success_spec.oracle.target_object != target.name:
        print(f"[WARN] oracle target_object {success_spec.oracle.target_object!r} "
              f"!= proposer object {target.name!r}")
        return False

    total_in = prop_usage.input_tokens + reward_usage.input_tokens
    total_out = prop_usage.output_tokens + reward_usage.output_tokens
    print(f"[totals] tokens in={total_in} out={total_out}")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="lift the cube")
    args = p.parse_args()
    ok = run(args.seed)
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
