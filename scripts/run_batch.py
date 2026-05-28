"""Run a batch of seeds through the M2 critic-aware pipeline.

    .venv/bin/python scripts/run_batch.py --seed-file seeds.txt \
                                          --dataset-name vs-m2 \
                                          --max-critic-iterations 3 \
                                          [--no-resume] [--limit 20]

Output goes to data/<dataset-name>/:
  batch_log.csv               one row per attempted task
  manifests/<task_id>.json    one per validated task
  raw/<task_id>.npz           one per validated task
  meta/info.json + data/...   LeRobot dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from worldwright.pipeline import run_batch
from worldwright.pipeline.batch import _read_seeds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-file", type=Path, required=True)
    p.add_argument("--dataset-name", default="vs-m2")
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    p.add_argument("--max-critic-iterations", type=int, default=3)
    p.add_argument("--no-resume", action="store_true",
                   help="Disable skip-already-completed; rerun every seed.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N seeds from the file.")
    args = p.parse_args()

    seeds = _read_seeds(args.seed_file)
    if args.limit is not None:
        seeds = seeds[:args.limit]

    console = Console()
    console.print(f"[bold]worldwright batch[/bold]  "
                  f"seeds={len(seeds)}  dataset={args.dataset_name}  "
                  f"max_critic={args.max_critic_iterations}  "
                  f"resume={not args.no_resume}")

    summary = run_batch(
        seeds=seeds,
        dataset_name=args.dataset_name,
        backend=args.backend,
        max_critic_iterations=args.max_critic_iterations,
        resume=not args.no_resume,
        log=console.print,
    )

    console.print()
    console.print(f"[bold]BATCH DONE[/bold]")
    console.print(f"  attempted: {summary.attempted}")
    console.print(f"  succeeded: {summary.succeeded}")
    console.print(f"  skipped:   {summary.skipped}")
    console.print(f"  wallclock: {summary.wallclock_s/60:.1f} min")
    if summary.attempted:
        rate = summary.succeeded / summary.attempted
        console.print(f"  valid_rate: {rate:.0%}")
    return 0 if summary.succeeded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
