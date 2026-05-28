"""Print the validity metrics dashboard for a batch dataset.

    .venv/bin/python scripts/report_metrics.py --dataset-name vs-m2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from worldwright.metrics import aggregate_validity, render_validity_table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="vs-m2")
    args = p.parse_args()

    dataset_root = Path("data") / args.dataset_name
    metrics = aggregate_validity(dataset_root)

    console = Console()
    console.print(render_validity_table(metrics, args.dataset_name))

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
