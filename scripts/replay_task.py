"""Replay a validated task from its manifest with camera recording on.

    .venv/bin/python scripts/replay_task.py \
        --manifest data/vs-m2/manifests/wright-lift-the-cube-6fa274ff19.json \
        --mp4 assets/vs-m2-lift-the-cube.mp4

Reads the persisted scene_code + success_code + oracle, rebuilds the scene
under Genesis Metal with a camera attached, executes the same oracle the
batch ran, saves an mp4 of the playback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from worldwright.engine import WorldwrightScene
from worldwright.task import OracleHint
from worldwright.utils import compile_callable
from worldwright.verifier import verify


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mp4", type=Path, required=True)
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--cam-pos", default="2.2,-1.4,1.2")
    p.add_argument("--cam-lookat", default="0.65,0.0,0.15")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    task = manifest["task"]
    scene_code = manifest["scene"]["scene_code"]
    success_code = manifest["success"]["success_code"]
    oracle = OracleHint.model_validate(manifest["success"]["oracle"])

    build_scene = compile_callable(
        scene_code, "build_scene", extra_globals={"np": np}
    )
    success_fn = compile_callable(
        success_code, "success", extra_globals={"np": np}
    )

    cam_pos = tuple(float(x) for x in args.cam_pos.split(","))
    cam_lookat = tuple(float(x) for x in args.cam_lookat.split(","))

    args.mp4.parent.mkdir(parents=True, exist_ok=True)

    with WorldwrightScene(sim_dt=0.01, backend=args.backend) as scene:
        build_scene(scene)
        scene.add_camera(
            name="cam0", pos=cam_pos, lookat=cam_lookat,
            res=(960, 720), fov=35,
        )
        scene.build()
        scene.start_recording("cam0")
        report = verify(scene, scene.franka, success_fn, oracle)
        scene.stop_recording("cam0", save_to_filename=str(args.mp4), fps=args.fps)

    intent = task.get("intent", "<unknown>")
    print(f"task:   {intent}")
    print(f"passed: {report.passed}  reason: {report.reason}  "
          f"steps: {len(report.trajectory)}")
    if report.terminal_state is not None:
        for name, obj in report.terminal_state.objects.items():
            print(f"  terminal {name}: pos={obj.pos.round(3).tolist()}")
    print(f"mp4:    {args.mp4}  ({args.mp4.stat().st_size//1024} KB)")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
