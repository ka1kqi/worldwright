# Risk-3 followup — proposed remediations

**Trigger:** sprint 3's measurement reported `false_positive_rate: 57.1%`, well above the 20% trigger from the sprint contract.

## Observed failure mechanism

4 of 7 nominally-unsolvable seeds came back `pipeline_success=true`:

| seed (gt=unsolvable) | observed pipeline | inferred root cause |
|---|---|---|
| `lift a 30 centimetre cube` | success | Proposer normalised to a small cube within the [0.03, 0.06] m size band from its SYSTEM prompt |
| `lift the cube at y=1.5` | success | Proposer clamped y into [-0.20, 0.20] |
| `lift the cube floating at z=0.5` | success | Proposer placed cube at `z = size_z / 2.0` per its prompt — "floating" was discarded |
| `lift the cube at y=-1.2 outside the workspace` | success | Proposer clamped y into [-0.20, 0.20] |

In each case the Proposer's SYSTEM prompt enforces a workspace clamp that silently "fixes" the adversarial seed, so the pipeline ends up validating a different (solvable) task than the one the user asked for. The pipeline is correct that the SANITISED task is solvable; the FP is that we report this as solvability of the ORIGINAL seed.

## Proposal: contact-stability post-grasp check

A contact-stability check addresses a DIFFERENT failure class — silent finger-cube slip during the lift phase (the M1.5-era "cube nudged sideways" pattern) — but it does NOT fix the observed Risk-3 FPs.

Even so, it's a useful robustness add. Sketch:

- **Where to attach:** `worldwright/verifier/verifier.py`'s `verify(...)` runs the oracle then evaluates `success_fn(state)` per step. Add a `_check_grasp_stability(trajectory)` helper that scans every step from `oracle.target_object`'s first grasp-phase contact onwards through the lift phase, asserting that the cube remains in continuous contact with both fingertips (any uninterrupted run of ≥10 steps without contact is a slip → mark `passed=False, reason=GRASP_LOST`).
- **What it needs from the engine:** `worldwright/engine/handles.py`'s `SceneState.contacts` field is currently always `[]` (TODO from M1). Populating it requires wiring `RigidEntity.get_contacts()` (Genesis API at `genesis/engine/entities/rigid_entity/rigid_entity.py:3975`) into `WorldwrightScene.state()` — non-trivial but already designed for in the type system. About 30 LOC in `scene.py`.
- **Where it fits in the oracle:** `worldwright/solver/oracle.py`'s `_run_phase` for `grasp` phase should record the contact ids it expects to remain stable — the verifier compares the trajectory's contacts against that expected set.
- **Verification:** add a unit test in `tests/test_verifier.py` that injects a fake trajectory where the cube starts in contact then loses it mid-lift; assert `verify(...)` returns `reason=GRASP_LOST`.

Doesn't fix the observed Risk-3 FPs but materially improves the failure surface for the M1.5-pattern silent-slip case.

## Proposal: TaskSpec-vs-seed sanity check (the RIGHT fix for Risk-3)

The observed FPs all stem from the Proposer normalising an adversarial seed without flagging the normalisation. The fix is a lightweight post-Proposer check:

- **Where to attach:** new function `worldwright/agents/proposer.py::audit_taskspec_against_seed(seed, task) -> list[str]` returning a list of normalisation flags.
- **Implementation:** regex-extract numeric constraints from the seed (e.g. `\b(\d+)\s*cent`, `y\s*=\s*([-\d.]+)`, `\b(\d+)\s*mill`, `floating|wall|outside`), compare against the corresponding TaskSpec fields (object sizes, y coords). If the seed asked for size > 0.06 m or position outside the workspace, return a normalisation flag.
- **What to do with the flags:** pipeline raises an `Unsolvable` PipelineResult instead of running the oracle. Sprint 1's `Unsolvable` patch path is already plumbed through `run_with_critic`; surfacing this from the Proposer is a one-call addition before `emit_scene(task)`.
- **Verification:** unit test with the 4 observed FP seeds, asserting each triggers an Unsolvable.

This fix is much smaller than the contact-stability work (~50 LOC + 4 tests, no engine changes) and directly addresses the 4 measured FPs. Worth doing first.

## Recommendation

For the next milestone (M3 or "robustness"):

1. **First:** TaskSpec-vs-seed sanity check. Directly addresses 4/4 observed FPs. ~1 day.
2. **Second:** Contact-stability check + contacts-on-SceneState plumbing. Addresses the M1.5 silent-slip class. ~3 days.

Neither is in scope for "near-term polish" — both are real engineering work that needs its own sprint.

The 57.1% FP rate is the headline finding from this sprint. It's not catastrophic — the pipeline IS correctly solving every task it claims to — but it means the reported `valid_rate` is a measurement of "the pipeline's chosen interpretation of the seed", not "the literal seed as stated". Worth surfacing in any portfolio/recruiting context with that caveat.
