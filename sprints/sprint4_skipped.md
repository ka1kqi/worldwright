# Sprint 4 — SKIPPED

**Reason:** budget exceeded.

The sprint 4 plan flagged itself as "OPTIONAL — skip if budget tight",
specifically: "harness should skip this sprint if Sprints 1–3 combined
exceed $7 of LLM spend or take more than 8 hours wallclock".

Actual cumulative cost across sprints 1–3:

  Sprint 1 (critic upgrade):       ~$2.30
  Sprint 2 (skill diversity):      ~$5.65   (iter 1: $4.45; iter 2: $1.20)
  Sprint 3 (risk-3 confusion mat): ~$2.40
  -----------------------------------------
  Total:                          ~$10.35   (well over the $7 trigger)

Sprint 4 (asset diversity via curated mesh primitives) is deferred. The
mesh-loading path in `WorldwrightScene.add_mesh` already exists in the
engine wrapper; what's left is the prompt-prep + smoke-batch work. That
belongs in M3 (cloud GPU + Nyx) where parallel batched evaluation makes
the per-task cost much lower.
