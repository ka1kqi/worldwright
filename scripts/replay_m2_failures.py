"""Replay the 10 verify-stage failures from data/vs-m2__batch/batch_log.csv
through the upgraded Critic (sprint-1: PatchSuccessThreshold + sharper
decision tree).

    .venv/bin/python scripts/replay_m2_failures.py

Output:
    data/vs-m2-critic-fix__batch/batch_log.csv   (canonical log, batch.py
                                                  sibling-dir convention)
    data/vs-m2-critic-fix/batch_log.csv          (copy at the contract-
                                                  specified path)
    data/vs-m2-critic-fix/manifests/...          (one per re-validated task)
    data/vs-m2-critic-fix/raw/...                (one per re-validated task)

Cost: 10 seeds. Each pipeline run is ~$0.06-0.15 in Anthropic tokens, so
expect $0.60-$1.50 worst case. Stay well under the sprint $2 cap.

Bump --max-critic-iterations from 3 to 4 if the first replay yields <6/10
passes; the sprint contract permits that single bump and re-run.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from rich.console import Console

from worldwright.pipeline import run_batch


SOURCE_LOG = Path("data/vs-m2__batch/batch_log.csv")
DATASET_NAME = "vs-m2-critic-fix"
SIBLING_LOG = Path(f"data/{DATASET_NAME}__batch/batch_log.csv")
COPY_LOG = Path(f"data/{DATASET_NAME}/batch_log.csv")


def extract_failed_seeds(source_log: Path) -> list[str]:
    """Pull every seed whose row has success=false from the M2 batch log."""
    seeds: list[str] = []
    with source_log.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("success", "").lower() == "false":
                seeds.append(row["seed"])
    return seeds


def count_successes(log_path: Path) -> tuple[int, int]:
    """Return (n_success, n_total) by reading the given batch_log.csv."""
    if not log_path.exists():
        return 0, 0
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n_total = len(rows)
    n_success = sum(1 for r in rows if r.get("success", "").lower() == "true")
    return n_success, n_total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-critic-iterations", type=int, default=3,
        help="Critic loop budget per seed. Bump to 4 only if the first "
             "replay yields <6/10 (sprint-1 fallback).",
    )
    p.add_argument(
        "--source-log", type=Path, default=SOURCE_LOG,
        help="The M2 batch log to mine for success=false rows.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Force a fresh run (default: skip seeds already passing in the "
             "vs-m2-critic-fix log).",
    )
    args = p.parse_args()

    console = Console()

    if not args.source_log.exists():
        console.print(f"[red]source log not found: {args.source_log}[/red]")
        return 1

    seeds = extract_failed_seeds(args.source_log)
    console.print(
        f"[bold]replay_m2_failures[/bold]  source={args.source_log}  "
        f"failed_seeds={len(seeds)}  max_critic_iter={args.max_critic_iterations}"
    )
    for s in seeds:
        console.print(f"  - {s!r}")

    if not seeds:
        console.print("[red]no failed seeds found in source log; nothing to do.[/red]")
        return 1

    summary = run_batch(
        seeds=seeds,
        dataset_name=DATASET_NAME,
        backend="metal",
        max_critic_iterations=args.max_critic_iterations,
        resume=not args.no_resume,
        log=console.print,
    )

    # Mirror the canonical sibling log to the contract-specified location.
    COPY_LOG.parent.mkdir(parents=True, exist_ok=True)
    if SIBLING_LOG.exists():
        shutil.copyfile(SIBLING_LOG, COPY_LOG)
        console.print(f"[batch] copied log -> {COPY_LOG}")

    n_pass, n_total = count_successes(SIBLING_LOG)
    console.print()
    console.print(f"[bold]REPLAY DONE[/bold]")
    console.print(f"  seeds_replayed: {len(seeds)}")
    console.print(f"  attempted:      {summary.attempted}")
    console.print(f"  skipped:        {summary.skipped}")
    console.print(f"  succeeded:      {summary.succeeded}")
    console.print(f"  csv_pass_rate:  {n_pass}/{n_total}")
    console.print(f"  wallclock:      {summary.wallclock_s / 60:.1f} min")

    # Sprint-1 acceptance bar: >= 6/10 pass.
    if n_pass >= 6:
        console.print(f"[green]PASS: {n_pass}/{n_total} >= 6/10 contract bar.[/green]")
        return 0
    console.print(
        f"[red]MISS: {n_pass}/{n_total} below 6/10 bar; "
        f"sprint fallback says bump max_critic_iterations to 4 and re-run.[/red]"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
