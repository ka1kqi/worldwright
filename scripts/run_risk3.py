"""Risk-3: solvability false-positive confusion matrix.

Reads `scripts/seeds_risk3.txt` (TSV: <seed>\\t<gt_label>), runs each seed
through the M2 critic-aware pipeline at `--max-critic-iterations 2` (override
of the default 4 to cap cost), then joins the pipeline's `success` flag
against the human-labelled gt and emits:

    data/risk3/cm.csv       — task_id,seed,gt_label,pipeline_success,pipeline_failure_stage
    data/risk3/summary.md   — confusion matrix counts + false_positive_rate: NN.N%

If FP-rate > 20%, the caller is expected to draft sprints/sprint3_followup.md.

Each pipeline run is spawned as a subprocess via scripts/run_single_task.py
(matches batch.py) to sidestep pyglet/cocoa display state leaking between
tasks under macOS Metal.

Usage:
    .venv/bin/python scripts/run_risk3.py
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from worldwright.pipeline.batch import LOG_FIELDS, _seed_to_task_id


VALID_LABELS = {"solvable", "borderline", "unsolvable"}


def _read_seeds_tsv(seed_file: Path) -> list[tuple[str, str]]:
    """Parse TSV; return list of (seed, gt_label). Skip blank/comment lines."""
    rows: list[tuple[str, str]] = []
    with seed_file.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "\t" not in line:
                raise ValueError(
                    f"seeds_risk3.txt line missing tab separator: {line!r}"
                )
            seed, label = line.split("\t", 1)
            seed = seed.strip()
            label = label.strip()
            if label not in VALID_LABELS:
                raise ValueError(
                    f"unknown gt_label {label!r} (expected one of {VALID_LABELS})"
                )
            rows.append((seed, label))
    return rows


def _run_one(
    *,
    seed: str,
    task_id: str,
    dataset_name: str,
    repo_id: str,
    max_critic_iterations: int,
    repo_root: Path,
) -> dict[str, str]:
    """Spawn run_single_task.py for one seed; return its parsed CSV row dict.

    On crash, returns a synthesised row with failure_stage=subprocess_crash.
    """
    python = repo_root / ".venv" / "bin" / "python"
    script = repo_root / "scripts" / "run_single_task.py"
    cmd = [
        str(python), str(script),
        "--seed", seed,
        "--task-id", task_id,
        "--dataset-name", dataset_name,
        "--repo-id", repo_id,
        "--backend", "metal",
        "--max-critic-iterations", str(max_critic_iterations),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(repo_root),
        env=os.environ,
    )
    stdout, _ = proc.communicate()
    lines = stdout.decode("utf-8").strip().splitlines()
    if not lines:
        return {
            **{k: "" for k in LOG_FIELDS},
            "task_id": task_id,
            "seed": seed,
            "success": "false",
            "failure_stage": "subprocess_crash",
            "failure_message": f"subprocess exit {proc.returncode}, no CSV output",
        }
    # run_single_task.py writes one CSV row (no header) to stdout.
    reader = csv.DictReader([",".join(LOG_FIELDS), lines[-1]])
    row = next(reader)
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-file", type=Path,
                   default=Path("scripts/seeds_risk3.txt"))
    p.add_argument("--dataset-name", default="vs-risk3")
    p.add_argument("--max-critic-iterations", type=int, default=2,
                   help="Override the pipeline default (4) to cap cost. "
                        "Unsolvable seeds otherwise burn the full critic loop.")
    p.add_argument("--out-dir", type=Path, default=Path("data/risk3"))
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    seed_rows = _read_seeds_tsv(args.seed_file)
    if len(seed_rows) > 20:
        print(f"[risk3] HARD CAP: seed file has {len(seed_rows)} rows (>20). "
              f"Refusing to run.", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = args.out_dir / "cm.csv"
    summary_path = args.out_dir / "summary.md"
    repo_id = f"ka1kqi/worldwright-{args.dataset_name}"

    print(f"[risk3] running {len(seed_rows)} seeds  "
          f"dataset={args.dataset_name}  max_critic={args.max_critic_iterations}")

    cm_rows: list[dict[str, str]] = []
    with cm_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "seed", "gt_label",
                        "pipeline_success", "pipeline_failure_stage"],
        )
        writer.writeheader()
        for i, (seed, gt_label) in enumerate(seed_rows, start=1):
            task_id = _seed_to_task_id(seed)
            print(f"[risk3] {i}/{len(seed_rows)}  gt={gt_label}  "
                  f"seed={seed!r}  task_id={task_id}")
            row = _run_one(
                seed=seed,
                task_id=task_id,
                dataset_name=args.dataset_name,
                repo_id=repo_id,
                max_critic_iterations=args.max_critic_iterations,
                repo_root=repo_root,
            )
            cm_row = {
                "task_id": task_id,
                "seed": seed,
                "gt_label": gt_label,
                "pipeline_success": row.get("success", "false"),
                "pipeline_failure_stage": row.get("failure_stage", "") or "",
            }
            writer.writerow(cm_row)
            f.flush()
            cm_rows.append(cm_row)
            print(f"[risk3]   -> pipeline_success={cm_row['pipeline_success']}  "
                  f"stage={cm_row['pipeline_failure_stage'] or '-'}")

    # ---- Confusion-matrix counts. ----
    def _pass(r: dict[str, str]) -> bool:
        return r["pipeline_success"].lower() == "true"

    tp = sum(1 for r in cm_rows if r["gt_label"] == "solvable" and _pass(r))
    fn = sum(1 for r in cm_rows if r["gt_label"] == "solvable" and not _pass(r))
    fp = sum(1 for r in cm_rows if r["gt_label"] == "unsolvable" and _pass(r))
    tn = sum(1 for r in cm_rows if r["gt_label"] == "unsolvable" and not _pass(r))
    borderline_pass = sum(
        1 for r in cm_rows if r["gt_label"] == "borderline" and _pass(r))
    borderline_fail = sum(
        1 for r in cm_rows if r["gt_label"] == "borderline" and not _pass(r))

    unsolvable_total = fp + tn
    fp_rate_pct = (fp / unsolvable_total * 100.0) if unsolvable_total else 0.0

    lines: list[str] = []
    lines.append("# Risk-3 — IK+plan_path solvability confusion matrix\n")
    lines.append(f"Total seeds run: {len(cm_rows)}  "
                 f"(dataset=`{args.dataset_name}`, "
                 f"max_critic_iterations={args.max_critic_iterations})\n")
    lines.append("## Confusion matrix (solvable vs unsolvable)\n")
    lines.append("| | pipeline=success | pipeline=fail |")
    lines.append("|---|---|---|")
    lines.append(f"| gt=solvable   | TP={tp} | FN={fn} |")
    lines.append(f"| gt=unsolvable | FP={fp} | TN={tn} |\n")
    lines.append("## Borderline (excluded from TP/FP/TN/FN)\n")
    lines.append(f"- borderline_pass = {borderline_pass}")
    lines.append(f"- borderline_fail = {borderline_fail}\n")
    lines.append("## False-positive rate\n")
    lines.append(f"FP / (FP + TN) = {fp} / {unsolvable_total}")
    lines.append(f"false_positive_rate: {fp_rate_pct:.1f}%\n")

    if fp_rate_pct > 20.0:
        lines.append("## Followup\n")
        lines.append("False-positive rate > 20%. See "
                     "`sprints/sprint3_followup.md` for the proposed "
                     "contact-stability post-grasp check.\n")
    else:
        lines.append("## Followup\n")
        lines.append("False-positive rate <= 20%: no followup required.\n")

    summary_path.write_text("\n".join(lines))
    print(f"\n[risk3] wrote {cm_path}")
    print(f"[risk3] wrote {summary_path}")
    print(f"[risk3] FP-rate = {fp_rate_pct:.1f}%  "
          f"(TP={tp} FP={fp} TN={tn} FN={fn} "
          f"borderline={borderline_pass + borderline_fail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
