"""Batch pipeline runner -- run many seeds in sequence, log + dataset persistence,
resume/skip, graceful shutdown.
"""

from __future__ import annotations

import csv
import hashlib
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from worldwright.pipeline.pipeline import PipelineResult, run_with_critic


LOG_FIELDS = [
    "task_id",
    "seed",
    "success",
    "failure_stage",
    "failure_message",
    "attempt",
    "critic_iterations",
    "wallclock_s",
    "oracle_steps",
    "tokens_in_total",
    "tokens_out_total",
    "lerobot_root",
    "manifest_path",
]


def _seed_to_task_id(seed: str) -> str:
    """Deterministic task_id derived from the seed text. Safe filename chars."""
    sanitized = re.sub(r"[^a-z0-9]+", "-", seed.lower())[:32].strip("-")
    digest = hashlib.sha256(seed.encode()).hexdigest()[:10]
    return f"wright-{sanitized}-{digest}"


def _read_seeds(seed_file: Path) -> list[str]:
    seeds: list[str] = []
    with seed_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(line)
    return seeds


def _completed_task_ids(log_path: Path) -> set[str]:
    """Read batch_log.csv (if any) and return the task_ids that already succeeded."""
    if not log_path.exists():
        return set()
    done: set[str] = set()
    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("success", "").lower() == "true":
                done.add(row["task_id"])
    return done


def _result_row(result: PipelineResult) -> dict[str, str]:
    m = result.metrics
    total_in = sum(u.input_tokens for u in m.tokens.values())
    total_out = sum(u.output_tokens for u in m.tokens.values())
    return {
        "task_id": result.task_id,
        "seed": result.seed,
        "success": "true" if result.success else "false",
        "failure_stage": result.failure_stage or "",
        "failure_message": (result.failure_message or "").replace("\n", " ")[:300],
        "attempt": str(result.attempt),
        "critic_iterations": str(m.critic_iterations),
        "wallclock_s": f"{m.wallclock_s:.1f}",
        "oracle_steps": str(m.oracle_steps),
        "tokens_in_total": str(total_in),
        "tokens_out_total": str(total_out),
        "lerobot_root": (result.paths or {}).get("lerobot_root", ""),
        "manifest_path": (result.paths or {}).get("manifest_path", ""),
    }


@dataclass
class BatchSummary:
    attempted: int = 0
    succeeded: int = 0
    skipped: int = 0
    wallclock_s: float = 0.0


def run_batch(
    seeds: Iterable[str],
    *,
    dataset_name: str = "vs-m2",
    backend: str = "metal",
    max_critic_iterations: int = 3,
    resume: bool = True,
    repo_id: str | None = None,
    log: Callable[[str], None] = print,
) -> BatchSummary:
    """Run each seed through run_with_critic, append a row to batch_log.csv.

    If ``resume`` is True (default), seeds whose deterministic task_id already
    appears in the log with success=true are skipped.
    """
    repo_id = repo_id or f"ka1kqi/worldwright-{dataset_name}"
    dataset_root = Path("data") / dataset_name
    # batch_log lives in a SIBLING dir so LeRobotDataset.create can claim
    # dataset_root cleanly (it refuses to create over an existing directory).
    log_dir = Path("data") / f"{dataset_name}__batch"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "batch_log.csv"
    done = _completed_task_ids(log_path) if resume else set()

    write_header = not log_path.exists()
    csv_file = log_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    summary = BatchSummary()
    t_start = time.time()

    stop_flag = {"stop": False}
    def _handle_sigint(signum, frame):  # noqa: ARG001
        log("\n[batch] SIGINT received -- finishing current task then exiting.")
        stop_flag["stop"] = True
    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        seeds_list = list(seeds)
        for i, seed in enumerate(seeds_list, start=1):
            task_id = _seed_to_task_id(seed)
            if task_id in done:
                log(f"[batch] {i}/{len(seeds_list)}  SKIP (already done)  task_id={task_id}")
                summary.skipped += 1
                continue
            log(f"\n[batch] {i}/{len(seeds_list)}  seed={seed!r}  task_id={task_id}")
            summary.attempted += 1
            try:
                result = run_with_critic(
                    seed=seed,
                    dataset_name=dataset_name,
                    backend=backend,
                    repo_id=repo_id,
                    task_id=task_id,
                    max_critic_iterations=max_critic_iterations,
                    log=log,
                )
            except Exception as e:  # noqa: BLE001
                log(f"[batch] task crashed: {type(e).__name__}: {e}")
                writer.writerow({
                    "task_id": task_id,
                    "seed": seed,
                    "success": "false",
                    "failure_stage": "crash",
                    "failure_message": f"{type(e).__name__}: {e}"[:300],
                    "attempt": "1",
                    "critic_iterations": "0",
                    "wallclock_s": "0.0",
                    "oracle_steps": "0",
                    "tokens_in_total": "0",
                    "tokens_out_total": "0",
                    "lerobot_root": "",
                    "manifest_path": "",
                })
                csv_file.flush()
                continue

            writer.writerow(_result_row(result))
            csv_file.flush()
            if result.success:
                summary.succeeded += 1
            log(f"[batch] result: success={result.success} "
                f"critic_iter={result.metrics.critic_iterations} "
                f"wallclock={result.metrics.wallclock_s:.0f}s")

            if stop_flag["stop"]:
                log("[batch] stopping after current task.")
                break
    finally:
        signal.signal(signal.SIGINT, old_handler)
        csv_file.close()

    summary.wallclock_s = time.time() - t_start
    log(f"\n[batch] DONE  attempted={summary.attempted}  succeeded={summary.succeeded}  "
        f"skipped={summary.skipped}  wallclock={summary.wallclock_s/60:.1f}min")
    return summary
