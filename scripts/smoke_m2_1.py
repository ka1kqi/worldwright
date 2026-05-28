"""M2.1 acceptance gate -- feed a hand-built failed PipelineResult to critique(),
assert the Critic returns a sensible patch (specifically a PatchOracle that
lowers reach_z or strengthens grasp force on the canonical 5-cm-cube failure).

    .venv/bin/python scripts/smoke_m2_1.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from worldwright.agents import PatchOracle, PatchSuccess, critique
from worldwright.task import (
    ObjectSpec,
    OracleHint,
    OraclePhase,
    SceneSpec,
    SuccessSpec,
    TaskSpec,
)


def _bad_taskspec() -> TaskSpec:
    return TaskSpec(
        seed="lift the cube",
        description="A 5 cm red cube on the table; lift it 15 cm.",
        intent="lift the red cube",
        objects=[
            ObjectSpec(name="red_cube", type="box", pos=(0.6, 0.0, 0.025),
                       size=(0.05, 0.05, 0.05), color=(1.0, 0.1, 0.1)),
        ],
        success_criteria="cube z > 0.15 m and held under gripper",
    )


def _bad_scene() -> SceneSpec:
    return SceneSpec(scene_code=(
        "def build_scene(scene):\n"
        "    scene.add_plane()\n"
        "    scene.add_franka()\n"
        "    scene.add_box(name='red_cube', size=(0.05,0.05,0.05), "
        "pos=(0.6,0.0,0.025), color=(1.0,0.1,0.1))\n"
    ))


def _bad_success() -> SuccessSpec:
    return SuccessSpec(
        success_code=(
            "import numpy as np\n"
            "def success(state):\n"
            "    cube = state.objects.get('red_cube')\n"
            "    if cube is None: return False\n"
            "    ee = state.ee_pos\n"
            "    xy_err = float(np.hypot(cube.pos[0]-ee[0], cube.pos[1]-ee[1]))\n"
            "    return bool(cube.pos[2] > 0.15 and xy_err < 0.05)\n"
        ),
        oracle=OracleHint(
            target_object="red_cube",
            phases=[
                OraclePhase(type="ik_pre_grasp", pos=(0.6, 0.0, 0.25), n_waypoints=200),
                OraclePhase(type="ik_reach",     pos=(0.6, 0.0, 0.13), steps=100),
                OraclePhase(type="grasp",        force=-0.5,           steps=100),
                OraclePhase(type="ik_lift",      pos=(0.6, 0.0, 0.28), steps=200),
            ],
        ),
    )


def _fake_failed_result():
    task = _bad_taskspec()
    scene = _bad_scene()
    success = _bad_success()
    terminal_state = SimpleNamespace(
        ee_pos=np.array([0.55, 0.0, 0.27]),
        ee_quat=np.array([0.0, 1.0, 0.0, 0.0]),
        objects={
            "red_cube": SimpleNamespace(
                pos=np.array([0.605, -0.044, 0.025]),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        },
    )
    vr = SimpleNamespace(
        reason="success_never_fired",
        message="success() returned False at terminal state",
        trajectory=[None] * 650,
        success_first_fired_at_step=None,
        terminal_state=terminal_state,
    )
    return SimpleNamespace(
        failure_stage="verify",
        failure_message="success_never_fired: success() returned False at terminal state",
        task=task,
        scene_spec=scene,
        success_spec=success,
        verifier_report=vr,
    )


def run() -> bool:
    result = _fake_failed_result()
    print("[critique] sending failed PipelineResult to Critic...")
    patch, usage = critique(result)
    print(f"[critique] tokens in={usage.input_tokens} out={usage.output_tokens}")
    print(f"[patch] kind={patch.kind}")
    print(f"[rationale] {getattr(patch, 'rationale', getattr(patch, 'reason', ''))}")

    if not isinstance(patch, (PatchOracle, PatchSuccess)):
        print(f"[FAIL] expected PatchOracle or PatchSuccess, got {type(patch).__name__}")
        return False

    new_oracle = patch.oracle
    print(f"[new_oracle] target={new_oracle.target_object} "
          f"phases={len(new_oracle.phases)}")
    for i, ph in enumerate(new_oracle.phases):
        print(f"  phase {i}: type={ph.type} pos={ph.pos} force={ph.force} "
              f"steps={ph.steps} n_waypoints={ph.n_waypoints}")

    # Acceptance: did the Critic actually fix something material?
    old_reach_z = 0.13
    new_reach = [ph for ph in new_oracle.phases if ph.type == "ik_reach"]
    if not new_reach:
        print("[FAIL] new oracle has no ik_reach phase")
        return False
    new_reach_z = new_reach[0].pos[2]
    new_grasp = [ph for ph in new_oracle.phases if ph.type == "grasp"]
    new_grasp_force = new_grasp[0].force if new_grasp else None

    changed_something = (
        new_reach_z < old_reach_z - 0.005      # lower reach
        or (new_grasp_force is not None
            and abs(new_grasp_force) > 0.5 + 0.05)  # stronger grasp
        or any(ph.type == "wait" for ph in new_oracle.phases)  # added wait
    )
    if not changed_something:
        print(f"[FAIL] Critic did not adjust reach_z, grasp force, or add wait. "
              f"reach_z={new_reach_z} grasp_force={new_grasp_force}")
        return False
    print(f"[OK] Critic changed something material: "
          f"reach_z {old_reach_z} -> {new_reach_z}, "
          f"grasp_force {-0.5} -> {new_grasp_force}, "
          f"has_wait={any(ph.type=='wait' for ph in new_oracle.phases)}")
    return True


def main() -> int:
    ok = run()
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
