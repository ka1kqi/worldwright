"""M2.2 acceptance gate -- run_with_critic recovers from a real LLM failure.

Uses the M1 sweet-spot-disabled regime by overriding the Proposer prompt with
a deliberately challenging seed that biases away from the sweet spot, so the
Critic actually has work to do.

    .venv/bin/python scripts/smoke_m2_2.py [--backend metal]
"""

from __future__ import annotations

import argparse
import sys

from worldwright.pipeline import run_with_critic


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="lift a 5 centimetre cube placed at x=0.60")
    p.add_argument("--dataset-name", default="vs-m2-smoke")
    p.add_argument("--backend", default="metal")
    p.add_argument("--max-critic-iterations", type=int, default=3)
    args = p.parse_args()

    result = run_with_critic(
        seed=args.seed,
        dataset_name=args.dataset_name,
        backend=args.backend,
        max_critic_iterations=args.max_critic_iterations,
    )

    print()
    print(f"[final] success={result.success} attempt={result.attempt}")
    print(f"[final] critic_iterations={result.metrics.critic_iterations}")
    print(f"[final] wallclock_s={result.metrics.wallclock_s:.1f}")
    print("[final] tokens per agent:")
    for k, u in result.metrics.tokens.items():
        print(f"    {k}: in={u.input_tokens} out={u.output_tokens}")
    if result.success:
        print(f"[final] artifacts: {result.paths}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
