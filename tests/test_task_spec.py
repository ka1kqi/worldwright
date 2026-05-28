"""Round-trip + validation tests for the task-spec Pydantic models."""

from __future__ import annotations

from worldwright.task import (
    ObjectSpec,
    OracleHint,
    OraclePhase,
    SceneSpec,
    SuccessSpec,
    TaskManifest,
    TaskMetrics,
    TaskSpec,
    TokenUsage,
)


def _sample_taskspec() -> TaskSpec:
    return TaskSpec(
        seed="franka-tabletop-block#a3f1",
        description="Pick up the small red cube and lift it 25 cm above the table.",
        intent="lift the red cube",
        objects=[
            ObjectSpec(
                name="cube_a",
                type="box",
                pos=(0.65, 0.0, 0.02),
                size=(0.04, 0.04, 0.04),
                color=(1.0, 0.0, 0.0),
            ),
        ],
        success_criteria="the cube is above z=0.15 m and still under the gripper",
    )


def test_taskspec_roundtrip() -> None:
    spec = _sample_taskspec()
    payload = spec.model_dump_json()
    restored = TaskSpec.model_validate_json(payload)
    assert restored == spec
    assert restored.objects[0].size == (0.04, 0.04, 0.04)


def test_full_manifest_roundtrip() -> None:
    task = _sample_taskspec()
    scene = SceneSpec(scene_code="def build_scene(scene): scene.add_plane()\n")
    success = SuccessSpec(
        success_code="def success(state): return state.objects['cube_a'].pos[2] > 0.15\n",
        oracle=OracleHint(
            target_object="cube_a",
            phases=[
                OraclePhase(type="ik_pre_grasp", pos=(0.65, 0.0, 0.25), n_waypoints=200),
                OraclePhase(type="ik_reach", pos=(0.65, 0.0, 0.13), steps=100),
                OraclePhase(type="grasp", force=-0.5, steps=100),
                OraclePhase(type="ik_lift", pos=(0.65, 0.0, 0.28), steps=200),
            ],
        ),
    )
    metrics = TaskMetrics(
        critic_iterations=1,
        tokens={"proposer": TokenUsage(input_tokens=1200, output_tokens=300)},
        wallclock_s=78.4,
        oracle_steps=600,
    )
    manifest = TaskManifest(
        task_id="wright-0000001",
        task=task,
        scene=scene,
        success=success,
        metrics=metrics,
        genesis_version="1.0.0",
    )

    payload = manifest.model_dump_json()
    restored = TaskManifest.model_validate_json(payload)
    assert restored == manifest
    assert restored.success.oracle.target_object == "cube_a"
    assert restored.metrics.tokens["proposer"].input_tokens == 1200
