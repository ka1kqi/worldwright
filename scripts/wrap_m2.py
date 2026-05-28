"""M2 wrap-up — run after the M2.7 batch completes.

Aggregates validity + diversity metrics for data/<dataset-name>/, writes
diversity.md, prints the validity dashboard, and writes a summary fragment
that the human can paste into DESIGN.html / README.

    .venv/bin/python scripts/wrap_m2.py --dataset-name vs-m2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from worldwright.metrics import (
    aggregate_validity,
    render_validity_table,
    write_diversity_report,
    diversity as diversity_fn,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="vs-m2")
    args = p.parse_args()

    dataset_root = Path("data") / args.dataset_name
    console = Console()

    # Validity
    console.rule("[bold]Validity metrics[/bold]")
    metrics = aggregate_validity(dataset_root)
    console.print(render_validity_table(metrics, args.dataset_name))

    # Diversity
    console.rule("[bold]Diversity report[/bold]")
    diversity_path = write_diversity_report(dataset_root, args.dataset_name)
    console.print(f"diversity.md written -> {diversity_path}")
    rpt = diversity_fn(dataset_root)
    console.print(f"  n_tasks                {rpt.n_tasks}")
    console.print(f"  object_type_entropy    {rpt.object_type_entropy:.3f} bits")
    console.print(f"  spatial_entropy        {rpt.spatial_entropy:.3f} bits")
    console.print(f"  unique colours         {len(rpt.color_counts)}")
    console.print(f"  unique grid cells      {len(rpt.grid_occupancy)}")
    console.print(f"  skill tags             {sorted(rpt.skill_counts)}")

    # Summary fragment
    console.rule("[bold]M2 acceptance summary fragment[/bold]")
    summary = f"""
## M2 results — {args.dataset_name}

- Attempted:        {metrics.n_attempted}
- Validated:        {metrics.n_succeeded}
- valid_rate:       {metrics.valid_rate:.0%}
- first_try_rate:   {metrics.first_try_success_rate:.0%}
  (validated tasks that needed ZERO Critic iterations)
- mean critic iters per validated:  {metrics.mean_critic_iterations_validated:.2f}
- mean wallclock per validated:     {metrics.mean_wallclock_s_validated:.1f}s
- mean tokens per validated:        in={metrics.mean_tokens_in_per_validated:.0f}
                                    out={metrics.mean_tokens_out_per_validated:.0f}

### Diversity
- Validated tasks:           {rpt.n_tasks}
- Object type entropy:       {rpt.object_type_entropy:.3f} bits
- Spatial layout entropy:    {rpt.spatial_entropy:.3f} bits (8x8 grid)
- Unique colours observed:   {len(rpt.color_counts)}
- Unique grid cells used:    {len(rpt.grid_occupancy)}/64
- Skill tags:                {", ".join(sorted(rpt.skill_counts))}

### Failure breakdown
""".strip()
    for stage, n in sorted(metrics.failure_reason_counts.items(),
                            key=lambda x: -x[1]):
        summary += f"\n- {stage}: {n}"

    summary_path = dataset_root / "M2_SUMMARY.md"
    summary_path.write_text(summary + "\n")
    console.print(summary)
    console.print(f"\n[dim]wrote {summary_path}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
