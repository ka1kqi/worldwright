"""Write the diversity markdown report for a dataset.

    .venv/bin/python scripts/report_diversity.py --dataset-name vs-m2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from worldwright.metrics import write_diversity_report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="vs-m2")
    args = p.parse_args()
    dataset_root = Path("data") / args.dataset_name
    out = write_diversity_report(dataset_root, args.dataset_name)
    print(f"wrote {out}")
    print()
    print(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
