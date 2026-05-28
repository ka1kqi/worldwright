"""Run a single task in a fresh process. Used by run_batch.py to sidestep
pyglet/cocoa display state leaking between tasks (IndexError on second-task
scene.build()).

    .venv/bin/python scripts/run_single_task.py --seed "lift the cube" \
        --dataset-name vs-m2 --task-id wright-... --max-critic-iterations 2

Writes a one-row CSV line to stdout (LOG_FIELDS schema, no header), so the
parent batch process can append it directly to batch_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys

from worldwright.pipeline import run_with_critic
from worldwright.pipeline.batch import LOG_FIELDS, _result_row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--backend", default="metal")
    p.add_argument("--max-critic-iterations", type=int, default=2)
    args = p.parse_args()

    # Silence the per-stage log lines on stdout (we only emit the CSV row).
    def _noop(*a, **k): pass

    try:
        result = run_with_critic(
            seed=args.seed,
            dataset_name=args.dataset_name,
            backend=args.backend,
            repo_id=args.repo_id,
            task_id=args.task_id,
            max_critic_iterations=args.max_critic_iterations,
            log=_noop,
        )
    except Exception as e:  # noqa: BLE001
        row = {k: "" for k in LOG_FIELDS}
        row["task_id"] = args.task_id
        row["seed"] = args.seed
        row["success"] = "false"
        row["failure_stage"] = "crash"
        row["failure_message"] = f"{type(e).__name__}: {e}"[:300]
        row["attempt"] = "1"
        row["critic_iterations"] = "0"
        for n in ("wallclock_s", "oracle_steps", "tokens_in_total", "tokens_out_total"):
            row[n] = "0"
        writer = csv.DictWriter(sys.stdout, fieldnames=LOG_FIELDS)
        writer.writerow(row)
        return 1

    writer = csv.DictWriter(sys.stdout, fieldnames=LOG_FIELDS)
    writer.writerow(_result_row(result))
    return 0 if result.success else 2  # 0=pass, 2=fail (not crash)


if __name__ == "__main__":
    sys.exit(main())
