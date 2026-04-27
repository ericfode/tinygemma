# Local Layer12 Phase Attribution Scope Fix

## Objective

Resolve `local-layer12-phase-attribution-runtime-probe-055` by determining whether the corrected `local-layer12` profiler signal is strong enough to justify a runtime experiment.

## Finding

Do not launch a runtime evo child from the pre-fix local-layer12 artifact.

Two independent reviews found that the attractive 885-source / ~14 ms `local-layer12` phase signal was not exclusive to local layer 12. The profiler captured under the requested phase-target scope, but later source attribution and original-capture summarization ran after the scope had reset to the default layer13 target. That made a local-layer12 artifact capable of reporting a layer13-shaped transitive closure as if it belonged to local-layer12.

## Change

`scripts/profile_decode_jit.py` now has `attribute_profile_execution_sources(execution_items, rows, phase_targets)`, which runs source attribution under the explicit phase-target scope used by the profile. `main()` also computes `original_capture.cache_write_phase_category_counts` under the same explicit scope.

The regression test `test_profile_source_attribution_uses_explicit_phase_targets_after_scope_reset` proves that a `local-layer12` target still attributes local-layer12 phase metadata correctly after the global scope has returned to its default.

## Corrected Artifact

Generated:

- `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700-scope-fixed.json`
- `benchmarks/gemma4-metal-decode-graph-local-layer12-phase-profile-700-scope-fixed.csv`

Key corrected values:

- `cache_write_phase_targets`: [`local-layer12`]
- MetalGraph batches preserved: 7
- raw gate/up runners: 0
- `source_attribution.status`: complete
- local-layer12 phase parent: 256 source items, ~3.255 ms, ~6.45% elapsed
- local-layer12 `kv_projection`: 250 source items, ~3.178 ms, ~6.30% elapsed
- conflict source count: 6
- unclassified source count: 0

## Decision

The corrected local-layer12 surface is real but small. It does not justify another immediate runtime patch: the obvious graphable K/V projection variants have already been tested or rejected, and the corrected hot slice is closer to a diagnostic target than a free optimization surface.

Use this artifact to choose a smaller future probe only if a new transformation is tied to the exclusive 256-source local-layer12 parent, not to the earlier 885-source transitive closure.

## Verification

- RED: `test_profile_source_attribution_uses_explicit_phase_targets_after_scope_reset` failed before `attribute_profile_execution_sources` existed.
- GREEN: targeted test passed.
- `tests/test_profile_decode_jit.py`: 34 passed, 2 warnings.
- Full suite: 83 passed, 2 warnings.
- METAL smoke: `rollout_jit_count=3`, `decode_fallback=False`.
