"""Validity metrics aggregator -- read batch_log.csv + manifests, compute
the M2 acceptance numbers, render as a Rich table.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidityMetrics:
    n_attempted: int = 0
    n_succeeded: int = 0
    n_skipped: int = 0
    valid_rate: float = 0.0

    first_try_success_rate: float = 0.0
    mean_critic_iterations_validated: float = 0.0
    p50_critic_iterations_validated: float = 0.0
    p90_critic_iterations_validated: float = 0.0

    mean_wallclock_s_validated: float = 0.0
    mean_oracle_steps_validated: float = 0.0
    mean_tokens_in_per_validated: float = 0.0
    mean_tokens_out_per_validated: float = 0.0

    failure_reason_counts: dict[str, int] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def aggregate_validity(dataset_root: Path | str) -> ValidityMetrics:
    dataset_root = Path(dataset_root)
    log_path = dataset_root / "batch_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"no batch_log.csv at {log_path}")

    rows: list[dict] = []
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    m = ValidityMetrics()
    m.rows = rows
    m.n_attempted = len(rows)
    m.n_succeeded = sum(1 for r in rows if r.get("success", "").lower() == "true")

    if m.n_attempted:
        m.valid_rate = m.n_succeeded / m.n_attempted

    validated = [r for r in rows if r.get("success", "").lower() == "true"]
    failed = [r for r in rows if r.get("success", "").lower() == "false"]

    if validated:
        crit_iters = [int(r.get("critic_iterations") or 0) for r in validated]
        wallclocks = [float(r.get("wallclock_s") or 0) for r in validated]
        oracle_steps = [int(r.get("oracle_steps") or 0) for r in validated]
        tok_in = [int(r.get("tokens_in_total") or 0) for r in validated]
        tok_out = [int(r.get("tokens_out_total") or 0) for r in validated]

        m.first_try_success_rate = sum(1 for c in crit_iters if c == 0) / len(crit_iters)
        m.mean_critic_iterations_validated = statistics.mean(crit_iters)
        crit_iters_sorted = sorted(crit_iters)
        m.p50_critic_iterations_validated = _percentile(crit_iters_sorted, 0.50)
        m.p90_critic_iterations_validated = _percentile(crit_iters_sorted, 0.90)

        m.mean_wallclock_s_validated = statistics.mean(wallclocks)
        m.mean_oracle_steps_validated = statistics.mean(oracle_steps)
        m.mean_tokens_in_per_validated = statistics.mean(tok_in)
        m.mean_tokens_out_per_validated = statistics.mean(tok_out)

    counts: dict[str, int] = {}
    for r in failed:
        stage = r.get("failure_stage") or "unknown"
        counts[stage] = counts.get(stage, 0) + 1
    m.failure_reason_counts = counts
    return m


def render_validity_table(m: ValidityMetrics, dataset_name: str = "") -> "Table":
    from rich.table import Table

    t = Table(title=f"M2 Validity Metrics  {dataset_name}".strip(),
              title_style="bold")
    t.add_column("metric", style="dim")
    t.add_column("value")

    t.add_row("attempted", str(m.n_attempted))
    t.add_row("succeeded", str(m.n_succeeded))
    t.add_row(
        "valid_rate",
        ("[bold green]" if m.valid_rate >= 0.5 else "[bold yellow]")
        + f"{m.valid_rate:.1%}" + "[/]",
    )
    t.add_row("first_try_success_rate (no critic)",
              f"{m.first_try_success_rate:.1%}")
    t.add_row("mean critic_iterations (validated)",
              f"{m.mean_critic_iterations_validated:.2f}")
    t.add_row("p50 / p90 critic_iterations",
              f"{m.p50_critic_iterations_validated:.0f} / "
              f"{m.p90_critic_iterations_validated:.0f}")
    t.add_row("mean wallclock_s (validated)",
              f"{m.mean_wallclock_s_validated:.1f}")
    t.add_row("mean oracle_steps (validated)",
              f"{m.mean_oracle_steps_validated:.0f}")
    t.add_row("mean tokens in/out (validated)",
              f"{m.mean_tokens_in_per_validated:.0f} / "
              f"{m.mean_tokens_out_per_validated:.0f}")
    if m.failure_reason_counts:
        t.add_row("",  "")
        t.add_row("failure breakdown", "")
        for stage, n in sorted(m.failure_reason_counts.items(),
                                key=lambda x: -x[1]):
            t.add_row(f"  {stage}", str(n))
    return t
