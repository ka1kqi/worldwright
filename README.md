# worldwright

**An LLM agent that autonomously authors simulation tasks in [Genesis World 1.0](https://github.com/Genesis-Embodied-AI/Genesis) — a modern reimplementation of [RoboGen](https://robogen-ai.github.io/) on the new physics stack.**

Given a domain seed, worldwright proposes a manipulation task, generates the scene as code, synthesises a success/reward function as code, runs it in Genesis World, verifies it's solvable with a scripted oracle, self-corrects failures, and logs validated tasks + trajectories as a training-ready [LeRobot](https://huggingface.co/docs/lerobot) dataset.

> **Status:** **M1 complete.** End-to-end LLM-driven pipeline runs on Apple M2 Metal: seed → Proposer (Sonnet) → SceneCoder (Sonnet) → RewardCoder (Opus) → Solver (Genesis IK + RRT-Connect) → Verifier → LeRobot dataset. Wallclock ~110 s / cost ~$0.10 per validated task. M2 (Critic loop + diversity) is next — see [milestones](https://github.com/ka1kqi/worldwright/milestones).

---

## Vertical-slice baseline

The video below is the human-baseline pick-and-lift that worldwright must eventually re-derive from natural-language seeds alone. Franka Panda + 4 cm cube, IK + RRT-Connect motion planning, force-controlled grasp. ~11 s, rendered at 960×720 on Apple M2 Metal.

[<video src="https://github.com/ka1kqi/worldwright/raw/main/assets/grasp_baseline.mp4" controls width="720"></video>](https://github.com/user-attachments/assets/4ef316c0-fa7e-4a45-aa9a-8302fc36bf9b)

▶ [Watch / download `grasp_baseline.mp4`](assets/grasp_baseline.mp4)

Reproduce it yourself:

```bash
.venv/bin/python scripts/reproduce_grasp.py --backend metal --record-mp4 assets/grasp_baseline.mp4
```

Exits 0 on success (`cube_z > 0.15` and held centred under the gripper); non-zero otherwise.

### LLM-driven end-to-end pipeline (M1 capstone)

```bash
.venv/bin/python scripts/run_pipeline.py --seed "lift the cube"
```

Wires Proposer → SceneCoder → RewardCoder → Solver → Verifier → LeRobot writer. On success, persists three artifacts under `data/<dataset-name>/`:

- LeRobot dataset (parquet, schema-checked, reloads via `LeRobotDataset(...)`)
- Raw NPZ shadow at `raw/<task_id>.npz` (no LeRobot dep on read)
- Full `TaskManifest` JSON at `manifests/<task_id>.json` (TaskSpec + scene_code + success_code + oracle + token + wallclock metrics)

Typical run: ~110 s wallclock, ~$0.10 in Anthropic tokens.

### Failure mode (M1.5 discovery)

When the Proposer chooses a 5 cm cube but the RewardCoder copy-pastes reach-z values from a 4 cm-cube worked example, the gripper descends too low relative to the cube centre and nudges it aside during grasp. The Verifier catches this cleanly with `passed=False reason=success_never_fired` and a full diagnostic terminal state — exactly the surface the Critic (M2) consumes. Deterministic reproduction:

https://github.com/user-attachments/assets/f2e0b1d7-1d41-4eeb-b8a2-cb3598393191

▶ [Watch `grasp_failure.mp4`](assets/grasp_failure.mp4) &middot; `.venv/bin/python scripts/record_failure_demo.py --record-mp4 assets/grasp_failure.mp4`

---

## Why this project

Genesis World 1.0 ships an excellent physics + rendering + motion-planning substrate, but the README explicitly places "agentic simulation, data generation" as a layer **above** the engine. There is no LLM hook, no reward DSL, no task-proposal scaffolding, no Objaverse / PartNet-Mobility loader, and no RoboGen-style pipeline in the repo (audited 2026-05-28; see Phase A in `DESIGN.html`).

worldwright is that missing layer, built on the **current** 1.0 API — not the pre-1.0 academic Genesis that the original RoboGen targeted.

---

## Architecture (one-line summary)

```
seed → Proposer → SceneCoder → RewardCoder → Solver(IK+plan_path) → Verifier → Critic↻ → LeRobot dataset
       (Sonnet)    (Sonnet)     (Opus)         (Genesis-native)                 (Sonnet)
```

Full architecture, data schema, milestones, and risk register live in [`DESIGN.html`](DESIGN.html). Open it in a browser:

```bash
open DESIGN.html
```

---

## Quickstart

Requires macOS (tested on Apple M2) or Linux + CUDA. Python 3.10–3.12 (3.12 recommended).

```bash
# clone Genesis World source for reference (optional, only needed for development)
git clone --depth 1 https://github.com/Genesis-Embodied-AI/Genesis.git ../genesis-world

# create venv and install
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install genesis-world "torch>=2.9.1"

# run the vertical-slice baseline (Metal on Apple Silicon, or pass --backend cpu / gpu)
.venv/bin/python scripts/reproduce_grasp.py --backend metal

# generate the demo video
.venv/bin/python scripts/reproduce_grasp.py --backend metal --record-mp4 assets/grasp_baseline.mp4
```

---

## Repo layout

```
worldwright/
├── DESIGN.html                 # design doc — open in browser
├── README.md                   # this file
├── pyproject.toml
├── scripts/
│   └── reproduce_grasp.py      # human-baseline vertical slice
├── src/worldwright/            # package scaffold (filling in during M1)
│   ├── engine/                 # only module that imports `genesis`
│   ├── task/                   # TaskSpec / SceneSpec / SuccessSpec
│   ├── agents/                 # Proposer, SceneCoder, RewardCoder, Critic
│   ├── solver/                 # IK + plan_path oracle
│   ├── verifier/               # runs solver, applies success predicate
│   ├── dataset/                # LeRobot writer
│   ├── pipeline/               # orchestrator
│   └── utils/                  # AST guard + restricted exec
├── assets/                     # videos, images
└── data/                       # generated datasets (gitignored)
```

---

## Milestones

| | Outcome | Status |
|---|---|---|
| **M0** | Phase A research; Phase B design; vertical-slice baseline runs on M2 Metal | ✅ done |
| **M1** | End-to-end agentic loop on one seed: all four agents wired, one validated LeRobot episode on disk | ✅ done — 7/7 sub-tickets |
| **M2** | Generalise within Franka tabletop. ≥ 50 validated tasks, `valid_rate` ≥ 0.5, Critic loop active | |
| **M3** | Scale on cloud NVIDIA. ~1 K validated trajectories, Nyx rendering, published LeRobot dataset | |
| **M4** | Optional — train Diffusion Policy / ACT on the generated dataset; report success-rate-vs-N | |

---

## Stack

- **Physics + rendering**: [Genesis World 1.0](https://github.com/Genesis-Embodied-AI/Genesis) (Quadrants compiler → Metal/CUDA/CPU)
- **Agent layer**: Anthropic Claude (Sonnet 4.6 for proposer / scene / critic, Opus 4.7 for reward synthesis)
- **Dataset format**: [LeRobot](https://huggingface.co/docs/lerobot)
- **Solver**: Genesis-native IK + RRT-Connect (`plan_path`)
- **Env tooling**: [uv](https://docs.astral.sh/uv/), Python 3.12

---

## License

MIT — see [`LICENSE`](LICENSE).
