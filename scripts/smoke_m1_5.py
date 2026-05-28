"""M1.5 smoke test -- Proposer + SceneCoder + RewardCoder + Solver + Verifier
end-to-end. The full pipeline minus the dataset writer (which is M1.6) and the
orchestrator (M1.7).

    .venv/bin/python scripts/smoke_m1_5.py [--seed "lift the cube"] [--backend metal]

Acceptance (from issue #6): on the hard-coded seed, returns passed=True with
terminal cube z >= ~0.15.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from worldwright.agents import emit_reward, emit_scene, propose
from worldwright.engine import WorldwrightScene
from worldwright.utils import compile_callable
from worldwright.verifier import verify


def run(seed: str, backend: str) -> bool:
    print(f"[seed] {seed!r}")

    # 1-3. Agents
    t0 = time.time()
    task, prop_usage = propose(seed)
    print(f"[proposer]     {time.time() - t0:.1f}s "
          f"intent={task.intent!r} objects={[o.name for o in task.objects]}")

    t0 = time.time()
    scene_spec, sc_usage = emit_scene(task)
    print(f"[scene_coder]  {time.time() - t0:.1f}s")

    t0 = time.time()
    success_spec, rw_usage = emit_reward(task)
    print(f"[reward_coder] {time.time() - t0:.1f}s "
          f"oracle_phases={len(success_spec.oracle.phases)} "
          f"target={success_spec.oracle.target_object!r}")

    # 4. Compile generated code
    build_scene = compile_callable(
        scene_spec.scene_code, "build_scene", extra_globals={"np": np}
    )
    success_fn = compile_callable(
        success_spec.success_code, "success", extra_globals={"np": np}
    )

    # 5-6. Build scene + verify
    with WorldwrightScene(sim_dt=0.01, backend=backend) as scene:
        build_scene(scene)
        scene.build()
        init_state = scene.state()
        print(f"[build] objects={list(init_state.objects)} "
              f"ee_pos={init_state.ee_pos.round(3).tolist()}")

        t0 = time.time()
        report = verify(scene, scene.franka, success_fn, success_spec.oracle)
        print(f"[verify] {time.time() - t0:.1f}s passed={report.passed} "
              f"reason={report.reason} "
              f"steps={len(report.trajectory)} "
              f"first_fire={report.success_first_fired_at_step}")
        if report.message:
            print(f"[message] {report.message}")
        if report.terminal_state is not None:
            for name, obj in report.terminal_state.objects.items():
                print(f"  terminal {name}: pos={obj.pos.round(3).tolist()}")
            ee = report.terminal_state.ee_pos
            print(f"  terminal ee_pos: {ee.round(3).tolist()}")

    total_in = prop_usage.input_tokens + sc_usage.input_tokens + rw_usage.input_tokens
    total_out = prop_usage.output_tokens + sc_usage.output_tokens + rw_usage.output_tokens
    print(f"[totals] tokens in={total_in} out={total_out}")

    return report.passed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="lift the cube")
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    args = p.parse_args()
    ok = run(args.seed, args.backend)
    print("[verdict]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
