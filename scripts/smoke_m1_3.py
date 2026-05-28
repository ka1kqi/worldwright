"""M1.3 smoke test — run Proposer + SceneCoder against the hard-coded seed,
sandbox-compile the emitted scene_code, and build it on a real Genesis scene.

Requires ANTHROPIC_API_KEY in the environment (or a .env file).

    .venv/bin/python scripts/smoke_m1_3.py [--seed "lift the cube"]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from worldwright.agents import emit_scene, propose
from worldwright.engine import WorldwrightScene
from worldwright.utils import compile_callable


def run(seed: str) -> bool:
    print(f"[seed] {seed!r}")

    # 1. Proposer
    t0 = time.time()
    task, prop_usage = propose(seed)
    print(f"[proposer] {time.time() - t0:.1f}s  "
          f"tokens in={prop_usage.input_tokens} out={prop_usage.output_tokens}")
    print(f"[task] description={task.description!r}")
    print(f"[task] intent={task.intent!r}")
    print(f"[task] objects={[o.name for o in task.objects]}")
    print(f"[task] success_criteria={task.success_criteria!r}")

    box_objects = [o for o in task.objects if o.type == "box"]
    if not box_objects:
        print("[FAIL] Proposer emitted no box objects")
        return False

    # 2. SceneCoder
    t0 = time.time()
    scene_spec, scene_usage = emit_scene(task)
    print(f"[scene_coder] {time.time() - t0:.1f}s  "
          f"tokens in={scene_usage.input_tokens} out={scene_usage.output_tokens}")
    print(f"[scene_code]\n{scene_spec.scene_code}")

    # 3. Sandbox-compile
    build_scene = compile_callable(
        scene_spec.scene_code,
        "build_scene",
        extra_globals={"np": np},
    )
    print("[sandbox] scene_code compiled cleanly")

    # 4. Actually build the scene in Genesis
    with WorldwrightScene(sim_dt=0.01, backend="metal") as scene:
        build_scene(scene)
        scene.build()
        state = scene.state()
        print(f"[scene] built. objects={list(state.objects.keys())}")
        print(f"[scene] franka_q={state.franka_q.round(2).tolist()}")
        # Sanity: at least one object placed, franka present.
        if not state.objects:
            print("[FAIL] No objects placed in scene")
            return False
        for name, obj in state.objects.items():
            print(f"  {name}: pos={obj.pos.round(3).tolist()}")

    total_in = prop_usage.input_tokens + scene_usage.input_tokens
    total_out = prop_usage.output_tokens + scene_usage.output_tokens
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
