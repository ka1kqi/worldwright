"""M1 capstone CLI -- run the full LLM-driven pipeline on one seed.

    .venv/bin/python scripts/run_pipeline.py [--seed "lift the cube"]
                                              [--dataset-name vs-m1]
                                              [--backend metal]
                                              [--max-attempts 3]
                                              [--task-id wright-xxxx]
                                              [--viewer]

Exits 0 if at least one attempt produced a validated task on disk; non-zero
otherwise. The final PipelineResult is printed as a Rich table.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from worldwright.pipeline import run_with_retries


def _result_table(result, max_attempts: int) -> Table:
    t = Table(title=f"PipelineResult  ({result.task_id})", title_style="bold")
    t.add_column("field", style="dim")
    t.add_column("value")

    t.add_row("seed", repr(result.seed))
    t.add_row("attempt", f"{result.attempt} / {max_attempts}")
    t.add_row(
        "success",
        "[bold green]TRUE[/bold green]" if result.success
        else "[bold red]FALSE[/bold red]",
    )
    if not result.success:
        t.add_row("failure_stage", str(result.failure_stage))
        t.add_row("failure_message", result.failure_message or "")

    if result.task is not None:
        t.add_row("intent", result.task.intent)
        t.add_row(
            "objects",
            ", ".join(f"{o.name}({o.type})" for o in result.task.objects),
        )

    if result.verifier_report is not None:
        rep = result.verifier_report
        t.add_row("oracle_phases", str(len(rep.oracle_result.trajectory) and
                                       len(result.success_spec.oracle.phases)))
        t.add_row("trajectory_steps", str(len(rep.trajectory)))
        t.add_row("success_first_fired_at_step",
                  str(rep.success_first_fired_at_step))
        if rep.terminal_state is not None:
            for name, obj in rep.terminal_state.objects.items():
                t.add_row(f"terminal[{name}].pos",
                          ", ".join(f"{x:.3f}" for x in obj.pos))

    m = result.metrics
    t.add_row("wallclock_s", f"{m.wallclock_s:.1f}")
    t.add_row("oracle_steps", str(m.oracle_steps))
    total_in = sum(u.input_tokens for u in m.tokens.values())
    total_out = sum(u.output_tokens for u in m.tokens.values())
    t.add_row("tokens (total)", f"in={total_in}  out={total_out}")
    for agent, usage in m.tokens.items():
        t.add_row(f"  tokens.{agent}",
                  f"in={usage.input_tokens}  out={usage.output_tokens}")

    if result.paths:
        for k, v in result.paths.items():
            t.add_row(k, v)

    return t


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="lift the cube")
    p.add_argument("--dataset-name", default="vs-m1")
    p.add_argument("--backend", default="metal", choices=["metal", "cpu", "gpu"])
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--task-id", default=None)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--repo-id", default="ka1kqi/worldwright-vs-m1")
    args = p.parse_args()

    console = Console()
    console.print(f"[bold]worldwright pipeline[/bold]  seed={args.seed!r}  "
                  f"dataset={args.dataset_name}  backend={args.backend}  "
                  f"max_attempts={args.max_attempts}")

    result = run_with_retries(
        seed=args.seed,
        max_attempts=args.max_attempts,
        dataset_name=args.dataset_name,
        backend=args.backend,
        repo_id=args.repo_id,
        task_id=args.task_id,
        show_viewer=args.viewer,
        log=console.print,
    )

    console.print()
    console.print(_result_table(result, args.max_attempts))
    console.print()
    console.print("[bold]" + ("PASS" if result.success else "FAIL") + "[/bold]")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
