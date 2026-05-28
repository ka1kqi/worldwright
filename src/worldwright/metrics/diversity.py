"""Diversity report: read manifests, compute the M2 diversity metrics
(object counts, type entropy, size buckets, spatial-layout entropy, skill tags),
emit a markdown report.

Per DESIGN.html §7.2. Reward-function clustering is deferred to M3 (requires
an embedding API).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


SIZE_BUCKETS = [(0.030, 0.035), (0.035, 0.040), (0.040, 0.045),
                (0.045, 0.050), (0.050, 0.055), (0.055, 0.061)]


# Skill keywords mapped from the natural-language intent.
SKILL_KEYWORDS = {
    "lift":  [r"\blift\w*", r"\bhoist\w*", r"\braise\w*"],
    "pick":  [r"\bpick\w*", r"\bgrasp\w*"],
    "place": [r"\bplace\w*", r"\bput\b", r"\bset\b"],
    "stack": [r"\bstack\w*"],
    "push":  [r"\bpush\w*", r"\bslide\w*"],
}


@dataclass
class DiversityReport:
    n_tasks: int = 0
    object_count_distribution: dict[int, int] = field(default_factory=dict)
    object_type_counts: dict[str, int] = field(default_factory=dict)
    object_type_entropy: float = 0.0
    size_bucket_counts: dict[str, int] = field(default_factory=dict)
    color_counts: dict[str, int] = field(default_factory=dict)
    grid_occupancy: dict[str, int] = field(default_factory=dict)  # "row-col" -> count
    spatial_entropy: float = 0.0
    skill_counts: dict[str, int] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)


def _entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _bucket_for_size(size_max: float) -> str:
    for lo, hi in SIZE_BUCKETS:
        if lo <= size_max < hi:
            return f"{lo*100:.1f}-{hi*100:.1f}cm"
    return "other"


def _color_name(rgb: list[float] | None) -> str:
    """Coarse colour name from an RGB triple in [0,1]."""
    if not rgb or len(rgb) < 3:
        return "unknown"
    r, g, b = rgb[0], rgb[1], rgb[2]
    if r > 0.6 and g < 0.4 and b < 0.4:  return "red"
    if r < 0.4 and g < 0.4 and b > 0.6:  return "blue"
    if r < 0.4 and g > 0.6 and b < 0.4:  return "green"
    if r > 0.6 and g > 0.6 and b < 0.4:  return "yellow"
    if r > 0.6 and g > 0.3 and g < 0.6 and b < 0.3: return "orange"
    if r > 0.6 and g < 0.4 and b > 0.6:  return "magenta"
    if r < 0.4 and g > 0.6 and b > 0.6:  return "cyan"
    if r > 0.4 and g < 0.4 and b > 0.4:  return "purple"
    if r > 0.7 and g > 0.7 and b > 0.7:  return "white"
    if r < 0.3 and g < 0.3 and b < 0.3:  return "black"
    return "other"


def _grid_cell(x: float, y: float) -> str:
    """8x8 grid over the workspace x in [0.50, 0.80], y in [-0.30, 0.30]."""
    col = max(0, min(7, int((x - 0.50) / 0.30 * 8)))
    row = max(0, min(7, int((y + 0.30) / 0.60 * 8)))
    return f"r{row}-c{col}"


def _skills_from_intent(intent: str) -> list[str]:
    lower = intent.lower()
    out: list[str] = []
    for skill, patterns in SKILL_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, lower):
                out.append(skill)
                break
    return out or ["unknown"]


def diversity(dataset_root: Path | str) -> DiversityReport:
    dataset_root = Path(dataset_root)
    manifests_dir = dataset_root / "manifests"
    if not manifests_dir.exists():
        raise FileNotFoundError(f"no manifests/ at {manifests_dir}")

    rpt = DiversityReport()
    obj_count_dist: Counter[int] = Counter()
    obj_type_counts: Counter[str] = Counter()
    size_bucket_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    grid_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    seeds: list[str] = []

    for path in sorted(manifests_dir.glob("*.json")):
        manifest = json.loads(path.read_text())
        rpt.n_tasks += 1
        task = manifest["task"]
        seeds.append(task["seed"])
        objs = task.get("objects", [])
        obj_count_dist[len(objs)] += 1
        for obj in objs:
            obj_type_counts[obj.get("type", "unknown")] += 1
            size = obj.get("size") or []
            if size:
                size_max = max(size)
                size_bucket_counts[_bucket_for_size(size_max)] += 1
            color_counts[_color_name(obj.get("color"))] += 1
            pos = obj.get("pos") or [0, 0, 0]
            grid_counts[_grid_cell(pos[0], pos[1])] += 1
        for skill in _skills_from_intent(task.get("intent", "")):
            skill_counts[skill] += 1

    rpt.object_count_distribution = dict(obj_count_dist)
    rpt.object_type_counts = dict(obj_type_counts)
    rpt.object_type_entropy = _entropy(obj_type_counts)
    rpt.size_bucket_counts = dict(size_bucket_counts)
    rpt.color_counts = dict(color_counts)
    rpt.grid_occupancy = dict(grid_counts)
    rpt.spatial_entropy = _entropy(grid_counts)
    rpt.skill_counts = dict(skill_counts)
    rpt.seeds = seeds
    return rpt


def render_markdown(rpt: DiversityReport, dataset_name: str = "") -> str:
    out = [f"# Diversity report — `{dataset_name}`",
           "",
           f"**Validated tasks:** {rpt.n_tasks}",
           "",
           "## Object counts per task",
           "| count | n_tasks |",
           "|---|---|"]
    for k in sorted(rpt.object_count_distribution):
        out.append(f"| {k} | {rpt.object_count_distribution[k]} |")

    out += ["",
            f"## Object type entropy: **{rpt.object_type_entropy:.3f} bits**",
            "| type | count |",
            "|---|---|"]
    for k, v in sorted(rpt.object_type_counts.items(), key=lambda x: -x[1]):
        out.append(f"| {k} | {v} |")

    out += ["",
            "## Size buckets",
            "| bucket | count |",
            "|---|---|"]
    for k, v in sorted(rpt.size_bucket_counts.items()):
        out.append(f"| {k} | {v} |")

    out += ["",
            "## Colours",
            "| colour | count |",
            "|---|---|"]
    for k, v in sorted(rpt.color_counts.items(), key=lambda x: -x[1]):
        out.append(f"| {k} | {v} |")

    out += ["",
            f"## Spatial layout entropy: **{rpt.spatial_entropy:.3f} bits**",
            "(8x8 grid over x in [0.50, 0.80], y in [-0.30, 0.30])",
            "",
            "| cell | count |",
            "|---|---|"]
    for k, v in sorted(rpt.grid_occupancy.items(), key=lambda x: -x[1]):
        out.append(f"| {k} | {v} |")

    out += ["",
            "## Skill tags (from intent keyword match)",
            "| skill | count |",
            "|---|---|"]
    for k, v in sorted(rpt.skill_counts.items(), key=lambda x: -x[1]):
        out.append(f"| {k} | {v} |")

    return "\n".join(out) + "\n"


def write_diversity_report(
    dataset_root: Path | str, dataset_name: str | None = None
) -> Path:
    dataset_root = Path(dataset_root)
    rpt = diversity(dataset_root)
    md = render_markdown(rpt, dataset_name or dataset_root.name)
    out_path = dataset_root / "diversity.md"
    out_path.write_text(md)
    return out_path
